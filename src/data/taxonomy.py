"""问题类型分类：Music-AVQA 与 AVQA 两套口径。"""
from __future__ import annotations

from typing import Any, Dict, Tuple

# Music-AVQA 九类（与 PLE qtype_label 0–8 一致）
MAVQA_QTYPE2IDX: Dict[str, Dict[str, int]] = {
    'Audio': {'Counting': 0, 'Comparative': 1},
    'Visual': {'Counting': 2, 'Location': 3},
    'Audio-Visual': {
        'Existential': 4,
        'Counting': 5,
        'Location': 6,
        'Comparative': 7,
        'Temporal': 8,
    },
}

# Valor32k-AVQA：modality × category（val+test 共 17 类）
VALOR32K_QTYPE2IDX: Dict[str, Dict[str, int]] = {
    'audio': {
        'action': 0,
        'count': 1,
        'description': 2,
        'location': 3,
        'relative-position': 4,
        'temporal': 5,
    },
    'visual': {
        'action': 6,
        'count': 7,
        'description': 8,
        'location': 9,
        'relative-position': 10,
        'temporal': 11,
    },
    'audio-visual': {
        'action': 12,
        'count': 13,
        'description': 14,
        'location': 15,
        'temporal': 16,
    },
}

# AVQA 官方 question_relation × question_type
AVQA_QTYPE2IDX: Dict[str, Dict[str, int]] = {
    'Sound': {'Which': 0, 'Come From': 1, 'Happening': 2},
    'View': {'Which': 3, 'Happening': 4, 'Where': 5, 'Why': 6},
    'Both': {
        'Which': 7,
        'Come From': 8,
        'Happening': 9,
        'Where': 10,
        'Why': 11,
        'Before Next': 12,
        'When': 13,
        'Used For': 14,
    },
}

AVQA_RELATION_TO_MODAL = {
    'Sound': 'Audio',
    'View': 'Visual',
    'Both': 'Audio-Visual',
}

# AVQA → Music-AVQA，供 PLE qtype_label 路由（模型仍按 9 类）
AVQA_TO_MAVQA_QTYPE: Dict[Tuple[str, str | None], Tuple[str, str]] = {
    ('Both', 'Which'): ('Audio-Visual', 'Comparative'),
    ('Both', 'Come From'): ('Audio-Visual', 'Location'),
    ('Both', 'Happening'): ('Audio-Visual', 'Existential'),
    ('Both', 'Where'): ('Audio-Visual', 'Location'),
    ('Both', 'Why'): ('Audio-Visual', 'Comparative'),
    ('Both', 'Before Next'): ('Audio-Visual', 'Temporal'),
    ('Both', 'When'): ('Audio-Visual', 'Temporal'),
    ('Both', 'Used For'): ('Audio-Visual', 'Existential'),
    ('Sound', 'Come From'): ('Audio', 'Comparative'),
    ('Sound', 'Which'): ('Audio', 'Comparative'),
    ('Sound', 'Happening'): ('Audio', 'Counting'),
    ('View', 'Which'): ('Visual', 'Location'),
    ('View', 'Happening'): ('Visual', 'Counting'),
    ('View', 'Where'): ('Visual', 'Location'),
    ('View', 'Why'): ('Visual', 'Location'),
}

AVQA_RELATION_FALLBACK = {
    'Sound': ('Audio', 'Comparative'),
    'View': ('Visual', 'Location'),
    'Both': ('Audio-Visual', 'Existential'),
}


def get_qtype_taxonomy_name(cfg: Any) -> str:
    return str(getattr(getattr(cfg, 'data', None), 'qtype_taxonomy', 'mavqa') or 'mavqa')


def get_qtype_taxonomy(cfg: Any) -> Dict[str, Dict[str, int]]:
    name = get_qtype_taxonomy_name(cfg)
    if name == 'avqa':
        return AVQA_QTYPE2IDX
    if name == 'valor32k':
        return VALOR32K_QTYPE2IDX
    return MAVQA_QTYPE2IDX


def parse_batched_qtype_pairs(taxonomy_name: str, batched_type) -> list[tuple[str, str]]:
    """从 collate 后的 type 字段解析每条样本的 (模态, 子类型)。"""
    name = str(taxonomy_name or 'mavqa')
    if name == 'valor32k':
        return list(zip(batched_type[1], batched_type[2]))
    return list(zip(batched_type[0], batched_type[1]))


def taxonomy_size(taxonomy: Dict[str, Dict[str, int]]) -> int:
    return sum(len(v) for v in taxonomy.values())


def taxonomy_gather_idx(
    taxonomy: Dict[str, Dict[str, int]],
    modal_type: str,
    qst_type: str,
) -> int | None:
    if modal_type not in taxonomy or qst_type not in taxonomy[modal_type]:
        return None
    offset = 0
    for modality, sub_map in taxonomy.items():
        for sub_type in sub_map:
            if modality == modal_type and sub_type == qst_type:
                return offset
            offset += 1
    return None


def avqa_to_mavqa_qtype(relation: str, qtype: str | None) -> Tuple[str, str]:
    key = (relation, qtype)
    if key in AVQA_TO_MAVQA_QTYPE:
        return AVQA_TO_MAVQA_QTYPE[key]
    if relation in AVQA_RELATION_FALLBACK:
        return AVQA_RELATION_FALLBACK[relation]
    return ('Audio-Visual', 'Existential')


def mavqa_qtype_label(modal: str, sub: str) -> int:
    return MAVQA_QTYPE2IDX[modal][sub]


VALOR32K_MODALITY_TO_MODAL = {
    "visual": "Visual",
    "audio": "Audio",
    "audio-visual": "Audio-Visual",
}


def valor32k_to_mavqa_qtype(modality: str, category: str) -> Tuple[str, str]:
    """Map valor32k modality/category to Music-AVQA 9-class PLE routing."""
    modal = VALOR32K_MODALITY_TO_MODAL.get(modality, "Audio-Visual")
    cat = category or "description"
    if cat == "count":
        if modal == "Audio":
            return modal, "Counting"
        if modal == "Visual":
            return modal, "Counting"
        return modal, "Counting"
    if cat in ("location", "relative-position"):
        if modal == "Audio":
            return modal, "Comparative"
        if modal == "Visual":
            return modal, "Location"
        return modal, "Location"
    if cat == "temporal":
        return "Audio-Visual", "Temporal"
    if modal == "Audio":
        return modal, "Comparative"
    if modal == "Visual":
        return modal, "Location"
    return modal, "Existential"
