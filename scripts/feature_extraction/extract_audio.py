"""AST audio feature extraction: frame stream + frequency-pooled patch tokens."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
FEAT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FEAT_DIR))

from pool_ast_patch import is_valid_pooled, pool_patch_numpy  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from transformers import ASTModel, AutoFeatureExtractor
    logger.info("使用 transformers 中的 AST 模型")
except ImportError as e:
    logger.error(f"无法导入 transformers: {e}")
    logger.error("请安装: pip install transformers")
    sys.exit(1)

SAMPLE_RATE = 16000


def log_exception(msg, exc):
    """记录详细异常信息"""
    import traceback
    logger.error(msg)
    logger.error(f"  异常类型: {type(exc).__name__}: {exc}")
    logger.error("  堆栈:")
    for line in traceback.format_exc().strip().split('\n'):
        logger.error(f"    {line}")


def extract_audio_from_video(video_path, output_wav_path, gpu_id="0", truncate_seconds=None):
    """
    从MP4视频中提取音频，保存为WAV文件
    """
    try:
        cmd = [
            'ffmpeg', '-y', '-loglevel', 'error',
        ]
        if truncate_seconds is not None:
            cmd.extend(['-t', str(truncate_seconds)])
        cmd.extend([
            '-i', str(video_path),
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', str(SAMPLE_RATE),
            '-ac', '1',
            str(output_wav_path),
        ])
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.warning(f"[GPU {gpu_id}] ffmpeg错误: {e}")
        logger.warning(f"  视频: {video_path}")
        if getattr(e, 'stderr', None):
            try:
                err = e.stderr.decode('ignore')[:500]
                logger.warning(f"  stderr: {err}")
            except Exception:
                pass
        return False
    except Exception as e:
        log_exception(f"[GPU {gpu_id}] 提取音频失败", e)
        return False


def extract_ast_features(wav_file, ast_model, feature_extractor, batch_size=32, gpu_id="0", max_seconds=None):
    """
    从 WAV 文件提取 AST 特征
    采样逻辑与 BEATs 相同：每秒一个特征（按秒切分音频，每秒 16000 采样点）

    Returns:
        cls_features: [T, 768] 最后一层 frame = (CLS + distillation) / 2，与 AST pooler_output 一致
        patch_last_features: [T, 1212, 768] 最后一层 patch tokens（不含 CLS/distillation）
        patch_mid_features: [T, 1212, 768] 倒数第三层 patch tokens
    """
    sr, snd = wavfile.read(wav_file)

    if len(snd.shape) > 1:
        audio_length_samples = snd.shape[0]
    else:
        audio_length_samples = len(snd)

    if max_seconds is not None:
        actual_duration_secs = int(max_seconds)
    else:
        actual_duration_secs = int(np.floor(audio_length_samples / sr))

    ch = 1
    L = sr * actual_duration_secs
    wav_data = np.zeros((L, ch))

    if len(snd.shape) > 1:
        snd = snd[:L, :]
    else:
        snd = snd[:L]
        snd = snd.reshape(-1, 1)

    wav_data = snd[:L, :]
    wav_data = wav_data / 32768.0

    T = actual_duration_secs
    device = next(ast_model.parameters()).device
    ast_model.eval()

    all_cls = []
    all_patch_last = []
    all_patch_mid = []

    with torch.no_grad():
        for batch_start in range(0, T, batch_size):
            batch_end = min(batch_start + batch_size, T)

            batch_audios = []
            for i in range(batch_start, batch_end):
                s = i * sr
                e = (i + 1) * sr
                if len(wav_data.shape) > 1:
                    data = wav_data[s:e, :]
                else:
                    data = wav_data[s:e]

                if data.shape[0] < sr:
                    pad_len = sr - data.shape[0]
                    pad = np.zeros((pad_len, data.shape[1] if len(data.shape) > 1 else 1))
                    if len(data.shape) > 1:
                        data = np.vstack([data, pad])
                    else:
                        data = np.hstack([data, pad.flatten()])

                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)

                batch_audios.append(data.astype(np.float32))

            inputs = feature_extractor(
                batch_audios,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
                return_attention_mask=False,
            )
            input_values = inputs.input_values.to(device)

            outputs = ast_model(input_values, output_hidden_states=True)
            hs = outputs.hidden_states

            last_layer = hs[-1]
            frame_batch = (
                (last_layer[:, 0, :] + last_layer[:, 1, :]) / 2
            ).cpu().numpy()
            patch_last_batch = last_layer[:, 2:, :].cpu().numpy()

            mid_layer = hs[-3]
            patch_mid_batch = mid_layer[:, 2:, :].cpu().numpy()

            all_cls.append(frame_batch)
            all_patch_last.append(patch_last_batch)
            all_patch_mid.append(patch_mid_batch)

    cls_features = np.concatenate(all_cls, axis=0)
    patch_last_features = np.concatenate(all_patch_last, axis=0)
    patch_mid_features = np.concatenate(all_patch_mid, axis=0)

    return cls_features, patch_last_features, patch_mid_features


def _npy_matches_target(path, expected_frames):
    if expected_frames is None:
        return True
    try:
        arr = np.load(path, mmap_mode="r")
        return int(arr.shape[0]) == int(expected_frames)
    except Exception:
        return False


def process_video(
    video_path,
    cls_output_dir,
    patch_last_output_dir,
    patch_mid_output_dir,
    ast_model,
    feature_extractor,
    batch_size,
    temp_audio_dir,
    gpu_id="0",
    target_frames=None,
    skip_patch_mid=False,
    patch_pooled_output_dir=None,
    save_pooled_only=False,
):
    """处理单个视频：提取音频 -> 提取特征 -> 保存"""
    video_name = video_path.stem
    cls_path = cls_output_dir / f'{video_name}.npy'
    patch_last_path = patch_last_output_dir / f'{video_name}.npy'
    patch_mid_path = patch_mid_output_dir / f'{video_name}.npy'
    pooled_path = (
        patch_pooled_output_dir / f'{video_name}.npy'
        if patch_pooled_output_dir is not None
        else None
    )

    if save_pooled_only and pooled_path is not None:
        outputs_ok = (
            cls_path.exists()
            and pooled_path.exists()
            and is_valid_pooled(pooled_path)
            and _npy_matches_target(cls_path, target_frames)
            and _npy_matches_target(pooled_path, target_frames)
        )
    else:
        outputs_ok = (
            cls_path.exists()
            and patch_last_path.exists()
            and _npy_matches_target(cls_path, target_frames)
            and _npy_matches_target(patch_last_path, target_frames)
        )
        if not skip_patch_mid:
            outputs_ok = outputs_ok and patch_mid_path.exists() and _npy_matches_target(
                patch_mid_path, target_frames
            )
    if outputs_ok:
        return True

    try:
        temp_wav_path = temp_audio_dir / f'{video_name}.wav'

        if not temp_wav_path.exists():
            if not extract_audio_from_video(
                video_path,
                temp_wav_path,
                gpu_id=gpu_id,
                truncate_seconds=target_frames,
            ):
                return False

        cls_features, patch_last_features, patch_mid_features = extract_ast_features(
            temp_wav_path,
            ast_model,
            feature_extractor,
            batch_size=batch_size,
            gpu_id=gpu_id,
            max_seconds=target_frames,
        )

        np.save(cls_path, cls_features)
        if save_pooled_only and pooled_path is not None:
            pooled = pool_patch_numpy(patch_last_features)
            try:
                pooled_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass
            np.save(pooled_path, pooled)
        else:
            np.save(patch_last_path, patch_last_features)
            if not skip_patch_mid:
                np.save(patch_mid_path, patch_mid_features)

        return True

    except Exception as e:
        log_exception(f"[GPU {gpu_id}] 处理视频 {video_name} 时出错", e)
        logger.error(f"  视频路径: {video_path}")
        return False


def find_optimal_batch_size(
    ast_model,
    feature_extractor,
    device,
    gpu_id="0",
    start_batch_size=8,
    max_batch_size=256,
):
    """自动寻找最优 batch_size"""
    logger.info(f"\n[GPU {gpu_id}] ==========================================")
    logger.info(f"[GPU {gpu_id}] 开始自动寻找最优 batch_size...")
    logger.info(f"[GPU {gpu_id}] ==========================================")

    current_batch_size = start_batch_size
    optimal_batch_size = start_batch_size
    test_audio = np.random.randn(SAMPLE_RATE).astype(np.float32)

    while current_batch_size <= max_batch_size:
        logger.info(f"[GPU {gpu_id}] 测试 batch_size = {current_batch_size}...")

        try:
            batch_audios = [test_audio for _ in range(current_batch_size)]
            inputs = feature_extractor(
                batch_audios,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
                return_attention_mask=False,
            )
            input_values = inputs.input_values.to(device)

            with torch.no_grad():
                ast_model.eval()
                outputs = ast_model(input_values)

            if outputs.last_hidden_state.shape[0] != current_batch_size:
                raise ValueError("输出 batch 大小不匹配")

            del outputs, input_values
            torch.cuda.empty_cache()
            import gc
            gc.collect()

            optimal_batch_size = current_batch_size
            logger.info("✓ 成功")

            if current_batch_size < 32:
                current_batch_size *= 2
            elif current_batch_size < 128:
                current_batch_size += 16
            else:
                current_batch_size += 32

        except torch.cuda.OutOfMemoryError:
            logger.info("✗ OOM")
            torch.cuda.empty_cache()
            break
        except Exception as e:
            logger.warning(f"✗ 错误: {e}")
            torch.cuda.empty_cache()
            break

    logger.info(f"\n[GPU {gpu_id}] 最优 batch_size: {optimal_batch_size}\n")
    return optimal_batch_size


def load_ast_model(model_path, device, gpu_id="0"):
    """从本地路径加载 AST 模型和特征提取器"""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"AST 模型目录不存在: {model_path}")

    logger.info(f"[GPU {gpu_id}] 加载 AST 模型: {model_path}")

    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            str(model_path), local_files_only=True
        )
        ast_model = ASTModel.from_pretrained(
            str(model_path), local_files_only=True
        )
        ast_model = ast_model.to(device)
        ast_model.eval()

        logger.info(f"[GPU {gpu_id}] ✓ AST 模型已加载到 {device}")
        return ast_model, feature_extractor

    except Exception as e:
        log_exception(f"[GPU {gpu_id}] 无法加载 AST 模型", e)
        logger.error(f"  模型路径: {model_path}")
        raise


def main():
    parser = argparse.ArgumentParser(description='AST 音频特征提取（Singularity 版本）')
    parser.add_argument('--video_list_file', type=str, default=None, help='视频列表文件（每行一个路径）')
    parser.add_argument('--video_dirs', type=str, nargs='+', help='视频目录列表')
    parser.add_argument('--output_dir', type=str, required=True, help='特征输出目录')
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID')
    parser.add_argument('--batch_size', type=int, default=None, help='batch 大小（None 表示自动寻找）')
    parser.add_argument('--no_auto_batch_size', action='store_true', help='禁用自动寻找 batch_size')
    parser.add_argument('--ast_model_path', type=str, default=None, help='AST 模型路径（默认 CKPT_DIR/ast）')
    parser.add_argument('--temp_audio_dir', type=str, default=None, help='临时音频目录')
    parser.add_argument(
        '--manifest', type=str, default=None,
        help='manifest.csv（含 video_id,target_frames），用于 T=floor(duration) 对齐',
    )
    parser.add_argument(
        '--skip_patch_mid', action='store_true',
        help='不保存 audio_ast_patch_mid',
    )
    parser.add_argument(
        '--save_pooled_only', action='store_true',
        help='内存 pool 后直接保存 audio_ast_patch_last_pooled，不落盘 patch_last',
    )

    args = parser.parse_args()
    if args.save_pooled_only:
        args.skip_patch_mid = True

    # 须由启动脚本在 python 进程前设置 CUDA_VISIBLE_DEVICES；勿用 --gpu_id 覆盖。
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        visible = str(args.gpu_id)
    log_gpu = visible.split(",")[0] if visible else str(args.gpu_id)
    args.gpu_id = log_gpu

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"[GPU {log_gpu}] 使用设备: {device} (CUDA_VISIBLE_DEVICES={visible})")

    CKPT_DIR = os.environ.get('CKPT_DIR', str(ROOT / 'ckpt'))

    if args.ast_model_path:
        ast_model_path = Path(args.ast_model_path)
    else:
        ast_model_path = Path(CKPT_DIR) / 'ast'

    if not ast_model_path.exists():
        logger.error(f"[GPU {args.gpu_id}] AST 模型目录不存在: {ast_model_path}")
        logger.error(f"  CKPT_DIR 环境变量: {os.environ.get('CKPT_DIR', '未设置')}")
        logger.error("请先下载模型到该目录，例如:")
        logger.error("  from transformers import ASTModel, AutoFeatureExtractor")
        logger.error("  m = ASTModel.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593')")
        logger.error(f"  m.save_pretrained('{ast_model_path}')")
        logger.error("  e = AutoFeatureExtractor.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593')")
        logger.error(f"  e.save_pretrained('{ast_model_path}')")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    cls_output_dir = output_dir / 'audio_ast_cls'
    patch_last_output_dir = output_dir / 'audio_ast_patch_last'
    patch_mid_output_dir = output_dir / 'audio_ast_patch_mid'
    patch_pooled_output_dir = output_dir / 'audio_ast_patch_last_pooled'

    cls_output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_pooled_only:
        try:
            patch_pooled_output_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
    if not args.save_pooled_only:
        patch_last_output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_patch_mid:
        patch_mid_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[GPU {args.gpu_id}] CLS (最后一层): {cls_output_dir}")
    if args.save_pooled_only:
        logger.info(f"[GPU {args.gpu_id}] Patch pooled: {patch_pooled_output_dir} [T, 12, 768]")
    else:
        logger.info(f"[GPU {args.gpu_id}] Patch 最后一层: {patch_last_output_dir}")
    if not args.skip_patch_mid:
        logger.info(f"[GPU {args.gpu_id}] Patch 倒数第三层: {patch_mid_output_dir}")

    if not shutil.which('ffmpeg'):
        logger.error("未找到 ffmpeg，请安装: apt-get install ffmpeg 或 conda install ffmpeg")
        logger.error(f"  PATH: {os.environ.get('PATH', '')[:200]}...")
        sys.exit(1)

    if args.temp_audio_dir:
        temp_audio_dir = Path(args.temp_audio_dir)
        temp_audio_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_audio_dir = Path(tempfile.mkdtemp(prefix=f'ast_audio_{args.gpu_id}_'))
        logger.info(f"[GPU {args.gpu_id}] 临时目录: {temp_audio_dir}")

    ast_model, feature_extractor = load_ast_model(ast_model_path, device, gpu_id=args.gpu_id)

    manifest_targets = {}
    if args.manifest and os.path.exists(args.manifest):
        import csv
        with open(args.manifest, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                manifest_targets[row["video_id"]] = int(row["target_frames"])
        logger.info(f"[GPU {args.gpu_id}] manifest targets: {len(manifest_targets)}")

    video_files = []

    if args.video_list_file and os.path.exists(args.video_list_file):
        with open(args.video_list_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    p = Path(line)
                    if p.exists() and p.suffix == '.mp4':
                        video_files.append(p)
        logger.info(f"[GPU {args.gpu_id}] 从文件读取 {len(video_files)} 个视频")
    elif args.video_dirs:
        for d in args.video_dirs:
            dp = Path(d)
            if dp.exists():
                files = list(dp.glob('*.mp4'))
                video_files.extend(files)
                logger.info(f"[GPU {args.gpu_id}] 从 {dp.name} 找到 {len(files)} 个 MP4")
    else:
        logger.error("必须提供 --video_list_file 或 --video_dirs")
        sys.exit(1)

    if not video_files:
        logger.error("未找到任何 MP4 文件")
        logger.error(f"  video_list_file: {args.video_list_file}")
        logger.error(f"  video_dirs: {args.video_dirs}")
        if args.video_list_file and os.path.exists(args.video_list_file):
            with open(args.video_list_file) as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            logger.error(f"  列表文件行数: {len(lines)}, 前3行: {lines[:3]}")
        sys.exit(1)

    auto_batch = (args.batch_size is None) and (not args.no_auto_batch_size)
    if auto_batch:
        batch_size = find_optimal_batch_size(
            ast_model, feature_extractor, device,
            gpu_id=args.gpu_id, start_batch_size=8, max_batch_size=256,
        )
    else:
        batch_size = args.batch_size or 32

    logger.info(f"[GPU {args.gpu_id}] batch_size: {batch_size}")

    success_count = 0
    skip_count = 0
    fail_count = 0

    for video_path in tqdm(video_files, desc=f"[GPU {args.gpu_id}] 提取 AST 特征"):
        video_name = video_path.stem
        target_frames = manifest_targets.get(video_name)
        cls_path = cls_output_dir / f'{video_name}.npy'
        patch_last_path = patch_last_output_dir / f'{video_name}.npy'
        patch_mid_path = patch_mid_output_dir / f'{video_name}.npy'
        pooled_path = patch_pooled_output_dir / f'{video_name}.npy'

        if args.save_pooled_only:
            skip_ok = (
                cls_path.exists()
                and pooled_path.exists()
                and is_valid_pooled(pooled_path)
                and _npy_matches_target(cls_path, target_frames)
                and _npy_matches_target(pooled_path, target_frames)
            )
        else:
            skip_ok = (
                cls_path.exists()
                and patch_last_path.exists()
                and _npy_matches_target(cls_path, target_frames)
                and _npy_matches_target(patch_last_path, target_frames)
            )
            if not args.skip_patch_mid:
                skip_ok = skip_ok and patch_mid_path.exists() and _npy_matches_target(
                    patch_mid_path, target_frames
                )
        if skip_ok:
            skip_count += 1
            continue

        if process_video(
            video_path,
            cls_output_dir,
            patch_last_output_dir,
            patch_mid_output_dir,
            ast_model,
            feature_extractor,
            batch_size,
            temp_audio_dir,
            args.gpu_id,
            target_frames=target_frames,
            skip_patch_mid=args.skip_patch_mid,
            patch_pooled_output_dir=patch_pooled_output_dir,
            save_pooled_only=args.save_pooled_only,
        ):
            success_count += 1
        else:
            fail_count += 1

    logger.info("\n" + "=" * 70)
    logger.info("处理完成！")
    logger.info("=" * 70)
    logger.info(f"成功: {success_count}, 跳过: {skip_count}, 失败: {fail_count}, 总计: {len(video_files)}")
    logger.info(f"CLS: {cls_output_dir} [T, 768]")
    if args.save_pooled_only:
        logger.info(f"Patch pooled: {patch_pooled_output_dir} [T, 12, 768]")
    else:
        logger.info(f"Patch 最后一层: {patch_last_output_dir} [T, num_patches, 768]")
        if not args.skip_patch_mid:
            logger.info(f"Patch 倒数第三层: {patch_mid_output_dir} [T, num_patches, 768]")

    if args.temp_audio_dir is None:
        logger.info(f"\n提示: 临时目录 {temp_audio_dir} 可手动删除")


if __name__ == '__main__':
    main()
