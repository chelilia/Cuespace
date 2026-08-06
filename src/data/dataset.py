import ast
import os
import torch
import json
import numpy as np
import logging

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from src.cuespace.layers.clip.clip import tokenize
from src.data.taxonomy import (
    AVQA_RELATION_TO_MODAL,
    MAVQA_QTYPE2IDX,
    VALOR32K_MODALITY_TO_MODAL,
    avqa_to_mavqa_qtype,
    mavqa_qtype_label,
    valor32k_to_mavqa_qtype,
)

logger = logging.getLogger("AVQA")


def normalize_question_content(question: str) -> str:
    q = question.lstrip().rstrip()
    if q.endswith("\uff1f"):
        q = q[:-1] + "?"
    return q


def _parse_templ_values(raw):
    if isinstance(raw, (list, tuple)):
        return list(raw)
    if raw is None or str(raw).strip() in ("", "[]"):
        return []
    parsed = ast.literal_eval(raw)
    if isinstance(parsed, (list, tuple)):
        return list(parsed)
    return [parsed]

qtype2idx = MAVQA_QTYPE2IDX

# Test release: fixed preprocessing; only feature paths vary per benchmark.
_IMG_SIZE = 336
_FRAME_SAMPLE_RATE = 1
_NUM_MCQ_OPTIONS = 4

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]

class AVQA_dataset(Dataset):

    def __init__(self, 
                 config: dict,
                 mode: str,
                 transform: transforms.Compose = None,
    ):
        self.mode = mode 
        self.config = config
        self.root = config.data.root
        
        self.audio_feat = (ROOT / self.root / config.data.audio_feat).as_posix() \
            if config.data.audio_feat is not None else None
        self.video_feat = (ROOT / self.root / config.data.video_feat).as_posix() \
            if config.data.video_feat is not None else None
        self.patch_feat = (ROOT / self.root / config.data.patch_feat).as_posix() \
            if config.data.patch_feat is not None else None
        self.visual_feat_format = getattr(config.data, 'visual_feat_format', None)
        self.audio_patch_feat = (ROOT / self.root / config.data.audio_patch_feat).as_posix() \
            if hasattr(config.data, 'audio_patch_feat') and config.data.audio_patch_feat is not None else None
        _mcq_text_feat = getattr(config.data, 'mcq_text_feat', None)
        self.mcq_text_feat = (ROOT / self.root / _mcq_text_feat).as_posix() \
            if _mcq_text_feat is not None else None
        
        self.quest_clap_feat = None

        self.tokenizer = tokenize
        self.size = _IMG_SIZE
        self.sample_rate = _FRAME_SAMPLE_RATE
        # 特征提取策略：1fps采样，不足 target_frames 的视频在 DataLoader 侧 padding
        self.target_frames = int(getattr(config.data, "target_frames", 60))
        self.answer_mode = str(getattr(config.data, 'answer_mode', 'open_vocab') or 'open_vocab')
        self.num_mcq_options = _NUM_MCQ_OPTIONS
        self.mcq_forward_mode = str(
            getattr(config.data, 'mcq_forward_mode', 'shared_stacked') or 'shared_stacked'
        )
        
        # 初始化transform
        self.transform = transform if transform is not None \
            else transforms.Compose([
                    transforms.Resize((self.size, self.size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN,
                                         std=IMAGENET_DEFAULT_STD),
                ])
        
        # 初始化logger
        from logging import getLogger
        logger = getLogger("AVQA")

        if self.mode != 'test':
            raise ValueError(f"AVQA_dataset is test-only; unsupported mode={self.mode!r}")

        annot_cfg = self.config.data.test_annot
        if isinstance(annot_cfg, (list, tuple)):
            if len(annot_cfg) == 0:
                raise ValueError("test_annot 为空列表，无法加载数据")
            annot_list = list(annot_cfg)
        else:
            annot_list = [annot_cfg]

        raw_samples = []
        for annot_rel in annot_list:
            annot_path = ROOT / self.root / annot_rel
            with open(file=annot_path.as_posix(), mode='r') as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                raw_samples.extend(loaded)
            else:
                raise ValueError(f"test_annot 文件内容不是 list: {annot_path}")

        self.samples = self.question_process(raw_samples)
        
        ans_quelen = self.get_max_question_length()
        self.answer_to_ix = ans_quelen['ans2ix']
        self.max_que_len = ans_quelen['max_que_len']
        if self.answer_mode == 'mcq':
            self.config.num_labels = self.num_mcq_options
        else:
            self.config.num_labels = len(self.answer_to_ix)
        
        video_list = []
        for sample in self.samples:
            video_name = sample['video_id']
            if video_name not in video_list:
                video_list.append(video_name)
        self.video_list = video_list
        self.video_len = 60 * len(video_list)#每个视频采样60帧
        self.cache = {}

    def pad_to_target_frames(self, tensor, target_frames=60):
        """
        将特征padding到目标帧数（重复最后一个token）
        
        Args:
            tensor: 输入tensor，形状为 [T, ...] 或 [T, N, ...]
            target_frames: 目标帧数，默认60
        
        Returns:
            padded_tensor: padding后的tensor，形状为 [target_frames, ...] 或 [target_frames, N, ...]
        """
        current_frames = tensor.shape[0]
        
        if current_frames >= target_frames:
            # 如果已经达到或超过目标帧数，直接截取
            return tensor[:target_frames]
        else:
            # 如果不足目标帧数，重复最后一个token来padding
            last_token = tensor[-1:].expand(target_frames - current_frames, *tensor.shape[1:])
            padded = torch.cat([tensor, last_token], dim=0)
            return padded
    
    def __len__(self):
        return len(self.samples)

    def _resolve_visual_feat_file(self, feat_root: str, video_id: str, feat_kind: str) -> Path:
        video_file = Path(feat_root) / f'{video_id}.npy'
        if video_file.exists():
            return video_file
        raise FileNotFoundError(f"{feat_kind} feature file not found: {video_file}")

    def _fill_question_from_template(self, template: str, templ_values_raw: str) -> str:
        question = template.split(' ')
        if question:
            last = question[-1]
            if last.endswith('?'):
                question[-1] = last[:-1]
            elif last.endswith('\uff1f'):
                question[-1] = last[:-1]
        templ_value_index = 0
        values = _parse_templ_values(templ_values_raw)
        for word_index in range(len(question)):
            if '<' in question[word_index]:
                question[word_index] = values[templ_value_index]
                templ_value_index += 1
        return ' '.join(question)

    @staticmethod
    def _mcq_candidate_text(question_content: str, option: str) -> str:
        q = question_content.rstrip().rstrip('?').strip()
        return f'{q} {option}'.strip()

    def load_samples(self, sample, index=None):
        # question preprocess
        if self.answer_mode == 'mcq':
            labels = torch.tensor([int(sample['answer'])], dtype=torch.long)
        else:
            labels = torch.tensor([self.answer_to_ix[sample['anser']]], dtype=torch.long)
        ques_type = ast.literal_eval(sample['type'])
        qtype_taxonomy = str(getattr(self.config.data, 'qtype_taxonomy', 'mavqa') or 'mavqa')
        if qtype_taxonomy == 'avqa':
            relation, avqa_qtype = ques_type[0], ques_type[1]
            mavqa_modal, mavqa_sub = avqa_to_mavqa_qtype(relation, avqa_qtype)
            qtype_label = torch.tensor([mavqa_qtype_label(mavqa_modal, mavqa_sub)], dtype=torch.long)
            modal_type = AVQA_RELATION_TO_MODAL.get(relation, 'Audio-Visual')
        elif qtype_taxonomy == 'valor32k':
            modality = ques_type[1] if len(ques_type) > 1 else 'audio-visual'
            category = ques_type[2] if len(ques_type) > 2 else 'description'
            mavqa_modal, mavqa_sub = valor32k_to_mavqa_qtype(modality, category)
            qtype_label = torch.tensor([mavqa_qtype_label(mavqa_modal, mavqa_sub)], dtype=torch.long)
            modal_type = VALOR32K_MODALITY_TO_MODAL.get(modality, 'Audio-Visual')
        else:
            qtype_label = torch.tensor([qtype2idx[ques_type[0]][ques_type[1]]], dtype=torch.long)
            modal_type = ques_type[0]
        cand_quest = None
        cand_words = None
        quest_words_offline = None
        if self.answer_mode == 'mcq' and self.mcq_text_feat is not None:
            quest_id = sample.get('question_id', None)
            if quest_id is None:
                raise ValueError(
                    f"MCQ 离线文本特征需要 question_id，video_id={sample.get('video_id')}"
                )
            feat_path = Path(self.mcq_text_feat) / f'{int(quest_id)}.npz'
            if not feat_path.exists():
                raise FileNotFoundError(f"MCQ 文本特征缺失: {feat_path}")
            with np.load(feat_path) as z:
                cand_quest = torch.from_numpy(np.asarray(z['cand_global'])).float()
                if self.mcq_forward_mode == 'per_option_full':
                    if 'cand_words' not in z:
                        raise KeyError(
                            f"per_option_full 需要 cand_words，缺失: {feat_path}"
                        )
                    cand_words = torch.from_numpy(np.asarray(z['cand_words'])).float()
                    quest = cand_quest[0]
                    quest_words_offline = None
                else:
                    quest = torch.from_numpy(np.asarray(z['quest_global'])).float()
                    quest_words_offline = torch.from_numpy(np.asarray(z['quest_words'])).float()
            qcr = None
            quest_clap = None
        else:
            # 在线 CLIP tokenize（MCQ 时 quest + cand 均在模型侧 encode）
            question = sample['question_content']
            quest = self.tokenizer(question, truncate=True).squeeze()
            qcr = None
            quest_clap = None

        if self.answer_mode == 'mcq' and cand_quest is None:
            options = sample['multi_choice']
            if len(options) != self.num_mcq_options:
                raise ValueError(
                    f"MCQ 需要 {self.num_mcq_options} 个选项，"
                    f"video_id={sample.get('video_id')} 得到 {len(options)}"
                )
            cand_rows = [
                self.tokenizer(
                    self._mcq_candidate_text(sample['question_content'], opt),
                    truncate=True,
                ).squeeze()
                for opt in options
            ]
            cand_quest = torch.stack(cand_rows, dim=0)
        
        # sampling frames
        name = sample['video_id']
        if self.video_feat is not None:
            if self.visual_feat_format == 'pstp_clip_b32':
                clip_file = self._resolve_visual_feat_file(
                    self.video_feat, name, 'PSTP-CLIP-B/32',
                )
                clip = np.load(clip_file)
                clip = torch.from_numpy(clip).float()
                if clip.ndim != 3 or clip.shape[1] < 2:
                    raise ValueError(
                        f"PSTP ViT-B/32 特征应为 [T, 1+N, D]，"
                        f"video_id={name} 得到 {tuple(clip.shape)}"
                    )
                video = clip[:, 0, :]
                patch = clip[:, 1:, :]
                video = self.pad_to_target_frames(video, self.target_frames)
                patch = self.pad_to_target_frames(patch, self.target_frames)
            else:
                video_file = self._resolve_visual_feat_file(
                    self.video_feat, name, 'CLIP',
                )
                video = np.load(video_file)
                video = torch.from_numpy(video).float()  # 确保转换为float32
                # 特征已经是1fps采样，不需要再采样
                # 如果不足60帧，padding到60（重复最后一个token）
                video = self.pad_to_target_frames(video, self.target_frames)
                patch = None

            if self.visual_feat_format != 'pstp_clip_b32' and self.patch_feat is not None:
                # 检查是否使用空的ToMe特征（用于极端测试）
                use_empty_tome = os.environ.get('USE_EMPTY_TOME', '').lower() in ('1', 'true', 'yes', 'on')
                empty_tome_mode = os.environ.get('EMPTY_TOME_MODE', 'zeros').lower()  # 'zeros', 'random', 'small_random'
                
                if use_empty_tome:
                    # 自动生成空的ToMe特征（维度匹配）
                    # ToMe特征维度: [num_frames, tokens=14, dim=1024]
                    # 注意：video已经padding到60帧，所以直接用video.shape[0]
                    num_frames = video.shape[0]  # 使用CLIP特征的帧数（已padding）
                    tokens = 14
                    dim = 1024
                    
                    if empty_tome_mode == 'zeros':
                        # 全零特征
                        patch = torch.zeros((num_frames, tokens, dim), dtype=torch.float32)
                    elif empty_tome_mode == 'random':
                        # 随机特征（标准正态分布）
                        patch = torch.randn((num_frames, tokens, dim), dtype=torch.float32)
                    elif empty_tome_mode == 'small_random':
                        # 小随机特征（标准差0.01）
                        patch = torch.randn((num_frames, tokens, dim), dtype=torch.float32) * 0.01
                    else:
                        # 默认使用全零
                        patch = torch.zeros((num_frames, tokens, dim), dtype=torch.float32)
                else:
                    patch_file = self._resolve_visual_feat_file(
                        self.patch_feat, name, 'ToMe',
                    )
                    patch = np.load(patch_file)
                    patch = torch.from_numpy(patch).float()  # 确保转换为float32
                    # 特征已经是1fps采样，不需要再采样
                    # 如果不足60帧，padding到60（重复最后一个token）
                    patch = self.pad_to_target_frames(patch, self.target_frames)
            elif self.visual_feat_format != 'pstp_clip_b32':
                patch = None
        else:
            frame_dir = ROOT / self.root / self.config.data.frames_dir / name
            frame_path = sorted(list(frame_dir.glob('*.jpg')))[:60] # some video processed wrong, having over 60 frames 
            frame_path = frame_path[::self.sample_rate]
            video = torch.stack([
                self.transform(Image.open(frame_path[i]).convert('RGB'))
                for i in range(len(frame_path))
            ], dim=0)
            patch = None
        
        # sampling audios
        if self.audio_feat is not None:
            # 双向 fallback 机制：尝试原始文件名和 _flip 版本
            audio_path = Path(self.audio_feat) / f'{name}.npy'
            if not audio_path.exists():
                # 尝试备用文件
                if name.endswith('_flip'):
                    # 如果 name 以 _flip 结尾，尝试去掉 _flip 的版本
                    fallback_name = name[:-5]  # 去掉 '_flip'
                    fallback_path = Path(self.audio_feat) / f'{fallback_name}.npy'
                    if fallback_path.exists():
                        audio_path = fallback_path
                    else:
                        # 记录缺失文件到日志
                        logger.warning(
                            f"[Dataset] 跳过样本 video_id={name} - "
                            f"Audio特征文件缺失: {audio_path} 和 {fallback_path} 都不存在"
                        )
                        raise FileNotFoundError(f"Audio feature file not found: {audio_path}")
                else:
                    # 如果 name 不以 _flip 结尾，尝试添加 _flip 的版本
                    fallback_path = Path(self.audio_feat) / f'{name}_flip.npy'
                    if fallback_path.exists():
                        audio_path = fallback_path
                    else:
                        # 记录缺失文件到日志
                        logger.warning(
                            f"[Dataset] 跳过样本 video_id={name} - "
                            f"Audio特征文件缺失: {audio_path} 和 {fallback_path} 都不存在"
                        )
                        raise FileNotFoundError(f"Audio feature file not found: {audio_path}")
            audio = np.load(audio_path.as_posix())
            audio = torch.from_numpy(audio).float()  # 确保转换为float32
            # 特征已经是1fps采样，不需要再采样
            # 如果不足60帧，padding到60（重复最后一个token）
            audio = self.pad_to_target_frames(audio, self.target_frames)
        else:
            audio_dir = ROOT / self.root / self.config.data.audios_dir
            audio_path = audio_dir / f'{name}.wav'
            audio = wavfile_to_examples(audio_path.as_posix(), num_secs=60)
            audio = torch.from_numpy(audio)
            # 如果从原始音频提取，wavfile_to_examples返回的是60秒的特征（1fps）
            # 如果不足60帧，padding到60（重复最后一个token）
            audio = self.pad_to_target_frames(audio, self.target_frames)
        
        # 加载音频patch特征（如果配置了）
        audio_patch = None
        if self.audio_patch_feat is not None:
            # 双向 fallback 机制：尝试原始文件名和 _flip 版本
            audio_patch_file = Path(self.audio_patch_feat) / f'{name}.npy'
            if not audio_patch_file.exists():
                # 尝试备用文件
                if name.endswith('_flip'):
                    # 如果 name 以 _flip 结尾，尝试去掉 _flip 的版本
                    fallback_name = name[:-5]  # 去掉 '_flip'
                    fallback_file = Path(self.audio_patch_feat) / f'{fallback_name}.npy'
                    if fallback_file.exists():
                        audio_patch_file = fallback_file
                    else:
                        # 记录缺失文件到日志
                        logger.warning(
                            f"[Dataset] 跳过样本 video_id={name} - "
                            f"Audio patch特征文件缺失: {audio_patch_file} 和 {fallback_file} 都不存在"
                        )
                        # 不抛出异常，允许audio_patch为None
                        audio_patch_file = None
                else:
                    # 如果 name 不以 _flip 结尾，尝试添加 _flip 的版本
                    fallback_file = Path(self.audio_patch_feat) / f'{name}_flip.npy'
                    if fallback_file.exists():
                        audio_patch_file = fallback_file
                    else:
                        # 记录缺失文件到日志
                        logger.warning(
                            f"[Dataset] 跳过样本 video_id={name} - "
                            f"Audio patch特征文件缺失: {audio_patch_file} 和 {fallback_file} 都不存在"
                        )
                        # 不抛出异常，允许audio_patch为None
                        audio_patch_file = None
            
            if audio_patch_file is not None and audio_patch_file.exists():
                audio_patch = np.load(audio_patch_file.as_posix())
                audio_patch = torch.from_numpy(audio_patch).float()  # 确保转换为float32
                # 特征已经是1fps采样，不需要再采样
                # 如果不足60帧，padding到60（重复最后一个token）
                audio_patch = self.pad_to_target_frames(audio_patch, self.target_frames)
        
        # 提取模态类型和任务ID（用于PLE多任务架构）
        task_id_map = {'Audio': 0, 'Visual': 1, 'Audio-Visual': 2}
        task_id = task_id_map.get(modal_type, 2)  # 默认使用Audio-Visual (2)
        
        _qid_raw = sample.get('question_id', None)
        question_id_tensor = (
            torch.tensor([int(_qid_raw)], dtype=torch.long)
            if _qid_raw is not None
            else torch.tensor([-1], dtype=torch.long)
        )

        data = {
            'quest': quest,
            'quest_clap': quest_clap,
            'qcr': None,
            'type': ques_type,
            'label': labels,
            'qtype_label': qtype_label,
            'video': video,
            'audio': audio,
            'name': name,
            'video_id': name,
            'question_content': sample['question_content'],
            'modal_type': modal_type,  # 添加模态类型
            'task_id': torch.tensor([task_id], dtype=torch.long),  # 添加任务ID (0=Audio, 1=Visual, 2=Audio-Visual)
            'patch': patch,       # 始终包含，None 时由 collate 处理
            'audio_patch': audio_patch,
            'question_id': question_id_tensor,
        }
        if cand_quest is not None:
            data['cand_quest'] = cand_quest
        if cand_words is not None:
            data['cand_words'] = cand_words
        if quest_words_offline is not None:
            data['quest_words'] = quest_words_offline
        if self.answer_mode == 'mcq':
            data['mcq_forward_mode'] = self.mcq_forward_mode
        return data

    def __getitem__(self, index):
        sample = self.samples[index]
        try:
            batch = self.load_samples(sample, index)
        except FileNotFoundError as e:
            # 记录缺失的样本并跳过，返回None让DataLoader过滤
            video_id = sample.get('video_id', 'N/A')
            logger.warning(f"[Dataset] 跳过样本 idx={index}, video_id={video_id} - {e}")
            return None
        
        return batch

    def question_process(self, samples):
        for index, sample in enumerate(samples):
            raw_q = normalize_question_content(sample['question_content'])
            samples[index]['question_content'] = self._fill_question_from_template(
                raw_q,
                sample['templ_values'],
            )
        return samples

    def get_max_question_length(self):
        ans_path = ROOT / self.root / self.config.data.ans_quelen
        if not ans_path.exists():
            raise FileNotFoundError(f"answer2idx 文件缺失: {ans_path}")
        with open(ans_path.as_posix(), mode='r') as f:
            return json.load(f)
