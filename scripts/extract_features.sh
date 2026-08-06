#!/usr/bin/env bash
# CueSpace offline feature extraction
#
# Produces the multi-level inputs used at inference (Sec. Multi-Level Input Representation):
#   • Visual frame stream            → frame_ViT-L14@336px/{video_id}.npy   [T, 768]
#   • Visual fine-grained tokens     → visual_tome14/{video_id}.npy         [T, 14, 1024]
#   • Audio frame stream             → ast/audio_ast_cls/{video_id}.npy     [T, 768]
#   • Audio fine-grained tokens      → ast/audio_ast_patch_last_pooled/...  [T, 12, 768]
#
# Usage:
#   # Full pipeline (mp4 → frames → visual + audio features)
#   bash scripts/extract_features.sh all \
#     --video-dir /path/to/mp4s \
#     --output ./data/feats/my_benchmark \
#     --gpu 0 --gpu1 1
#
#   # From existing JPEG frames only
#   bash scripts/extract_features.sh visual \
#     --frames ./data/raw_frames \
#     --output ./data/feats/my_benchmark \
#     --gpu 0 --gpu1 1
#
#   # Audio only (from mp4 list)
#   bash scripts/extract_features.sh audio \
#     --video-dir /path/to/mp4s \
#     --output ./data/feats/my_benchmark \
#     --gpu 0

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FEAT_DIR="${ROOT}/scripts/feature_extraction"
CKPT_DIR="${CKPT_DIR:-${ROOT}/ckpt}"
PYTHON="${PYTHON:-python3}"

STAGE="${1:-}"
shift || true

VIDEO_DIR=""
FRAMES_DIR=""
OUTPUT=""
GPU="0"
GPU1="1"
MAX_FRAMES="60"
BATCH_SIZE="128"
LIMIT=""
USE_ALL_FRAMES="0"
SINGLE_GPU="0"

usage() {
  sed -n '2,26p' "$0" | sed 's/^# \?//'
  echo ""
  echo "Stages: frames | visual | audio | all"
  echo ""
  echo "Options:"
  echo "  --video-dir DIR     MP4 directory (frames + audio stages)"
  echo "  --frames DIR        Existing frame root (visual stage)"
  echo "  --output DIR        Feature output root (required)"
  echo "  --gpu ID            Primary CUDA device (default 0)"
  echo "  --gpu1 ID           Secondary CUDA device for visual stage (default 1)"
  echo "  --single-gpu        Visual stage on --gpu only (disable parallel workers)"
  echo "  --max-frames N      Max frames per video (default 60)"
  echo "  --batch-size N      Visual batch size (default 128)"
  echo "  --use-all-frames    Keep variable T (Valor32k / AVQA style)"
  echo "  --limit N           Process only first N videos (smoke test)"
  exit 1
}

[[ -n "$STAGE" ]] || usage
[[ "$STAGE" == "help" || "$STAGE" == "-h" ]] && usage

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video-dir) VIDEO_DIR="$2"; shift 2 ;;
    --frames) FRAMES_DIR="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --gpu1) GPU1="$2"; shift 2 ;;
    --single-gpu) SINGLE_GPU=1; shift ;;
    --max-frames) MAX_FRAMES="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --use-all-frames) USE_ALL_FRAMES=1; shift ;;
    --limit) LIMIT="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

[[ -n "$OUTPUT" ]] || { echo "Error: --output is required"; exit 1; }

mkdir -p "$OUTPUT"
LOG_DIR="${OUTPUT}/logs"
mkdir -p "$LOG_DIR"

run_frames() {
  [[ -n "$VIDEO_DIR" ]] || { echo "frames stage needs --video-dir"; exit 1; }
  FRAMES_DIR="${FRAMES_DIR:-${OUTPUT}/frames}"
  mkdir -p "$FRAMES_DIR"
  extra=()
  [[ -n "$LIMIT" ]] && extra+=(--limit "$LIMIT")
  "$PYTHON" "${FEAT_DIR}/extract_frames.py" \
    --video-dir "$VIDEO_DIR" \
    --output-dir "$FRAMES_DIR" \
    --max-frames "$MAX_FRAMES" \
    "${extra[@]}"
}

run_visual() {
  FRAMES_DIR="${FRAMES_DIR:-${OUTPUT}/frames}"
  [[ -d "$FRAMES_DIR" ]] || { echo "visual stage needs existing --frames or run frames first"; exit 1; }

  FRAME_OUT="${OUTPUT}/frame_ViT-L14@336px"
  PATCH_OUT="${OUTPUT}/visual_tome14"
  mkdir -p "$FRAME_OUT" "$PATCH_OUT"

  EXTRACT_PY="${FEAT_DIR}/extract_visual.py"
  [[ -f "$EXTRACT_PY" ]] || { echo "Error: missing ${EXTRACT_PY}"; exit 1; }

  export CKPT_DIR="$CKPT_DIR"

  extra_visual=()
  [[ "$USE_ALL_FRAMES" == "1" ]] && extra_visual+=(--use_all_frames)

  if [[ "$SINGLE_GPU" == "0" ]]; then
    TMP="${OUTPUT}/tmp"
    mkdir -p "$TMP"
    mapfile -t DIRS < <(find "$FRAMES_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
    TOTAL=${#DIRS[@]}
    HALF=$((TOTAL / 2))
    LIST0="${TMP}/visual_gpu0.txt"
    LIST1="${TMP}/visual_gpu1.txt"
    printf '%s\n' "${DIRS[@]:0:${HALF}}" > "$LIST0"
    printf '%s\n' "${DIRS[@]:${HALF}}" > "$LIST1"
    echo "Visual extraction: ${TOTAL} videos on GPU ${GPU} (${HALF}) and GPU ${GPU1} ($((TOTAL - HALF)))"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$EXTRACT_PY" \
      --frames_base_dir "$FRAMES_DIR" \
      --frame_output_dir "$FRAME_OUT" \
      --patch_output_dir "$PATCH_OUT" \
      --video_list_file "$LIST0" \
      --batch_size "$BATCH_SIZE" \
      --num_frames "$MAX_FRAMES" \
      --gpu_id 0 \
      --ckpt_dir "$CKPT_DIR" \
      "${extra_visual[@]}" \
      > "${LOG_DIR}/visual_gpu0.log" 2>&1 &
    PID0=$!
    CUDA_VISIBLE_DEVICES="$GPU1" "$PYTHON" "$EXTRACT_PY" \
      --frames_base_dir "$FRAMES_DIR" \
      --frame_output_dir "$FRAME_OUT" \
      --patch_output_dir "$PATCH_OUT" \
      --video_list_file "$LIST1" \
      --batch_size "$BATCH_SIZE" \
      --num_frames "$MAX_FRAMES" \
      --gpu_id 0 \
      --ckpt_dir "$CKPT_DIR" \
      "${extra_visual[@]}" \
      > "${LOG_DIR}/visual_gpu1.log" 2>&1 &
    PID1=$!
    wait "$PID0" "$PID1"
  else
    echo "Visual extraction: single worker on GPU ${GPU}"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$EXTRACT_PY" \
      --frames_base_dir "$FRAMES_DIR" \
      --frame_output_dir "$FRAME_OUT" \
      --patch_output_dir "$PATCH_OUT" \
      --batch_size "$BATCH_SIZE" \
      --num_frames "$MAX_FRAMES" \
      --gpu_id 0 \
      --ckpt_dir "$CKPT_DIR" \
      "${extra_visual[@]}" \
      2>&1 | tee "${LOG_DIR}/visual.log"
  fi
  echo "Visual features → ${FRAME_OUT} , ${PATCH_OUT}"
}

run_audio() {
  [[ -n "$VIDEO_DIR" ]] || { echo "audio stage needs --video-dir"; exit 1; }

  AST_PY="${FEAT_DIR}/extract_audio.py"
  [[ -f "$AST_PY" ]] || { echo "Error: missing ${AST_PY}"; exit 1; }

  LIST="${OUTPUT}/tmp/video_list.txt"
  mkdir -p "$(dirname "$LIST")"
  "$PYTHON" "${FEAT_DIR}/build_video_list.py" --video-dir "$VIDEO_DIR" --output "$LIST"

  export CKPT_DIR="$CKPT_DIR"

  AST_OUT="${OUTPUT}/ast"
  mkdir -p "$AST_OUT"
  echo "Audio extraction on GPU ${GPU}"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$AST_PY" \
    --video_list_file "$LIST" \
    --output_dir "$AST_OUT" \
    --gpu_id 0 \
    --save_pooled_only \
    2>&1 | tee "${LOG_DIR}/audio.log"

  echo "Audio features → ${AST_OUT}/audio_ast_cls , ${AST_OUT}/audio_ast_patch_last_pooled"
}

case "$STAGE" in
  frames)
    run_frames
    ;;
  visual)
    run_visual
    ;;
  audio)
    run_audio
    ;;
  all)
    FRAMES_DIR="${OUTPUT}/frames"
    run_frames
    run_visual
    run_audio
    ;;
  *)
    echo "Unknown stage: $STAGE"
    usage
    ;;
esac

echo ""
echo "=========================================="
echo "Stage '${STAGE}' finished."
echo "Output root: ${OUTPUT}"
echo "Logs: ${LOG_DIR}"
echo ""
echo "Point configs/test_profiles.py feature paths to:"
echo "  video_feat  = <output>/frame_ViT-L14@336px"
echo "  patch_feat  = <output>/visual_tome14"
echo "  audio_feat  = <output>/ast/audio_ast_cls"
echo "  audio_patch_feat = <output>/ast/audio_ast_patch_last_pooled"
echo "=========================================="
