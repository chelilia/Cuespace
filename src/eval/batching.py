"""Test batch tensor assembly."""
from __future__ import annotations

import torch


def get_items(batch: dict, device: torch.device) -> dict:
    reshaped_data = dict(
        quest=batch['quest'],
        quest_clap=None,
        audio=batch['audio'].to(device).float(),
        video=batch['video'].to(device).float(),
        qtype_label=batch['qtype_label'].to(device).long().reshape(-1),
        patch=batch['patch'].to(device).float() if batch.get('patch') is not None else None,
        audio_patch=(
            batch['audio_patch'].to(device).float()
            if batch.get('audio_patch') is not None
            else None
        ),
        qcr=None,
        label=batch['label'].to(device).long().reshape(-1),
        task_id=batch['task_id'].to(device).long().squeeze(-1) if 'task_id' in batch else None,
        modal_type=batch.get('modal_type'),
        question_id=(
            batch['question_id'].to(device).long().reshape(-1)
            if batch.get('question_id') is not None
            else None
        ),
        video_id=batch.get('name'),
    )
    if batch.get('cand_quest') is not None:
        reshaped_data['cand_quest'] = batch['cand_quest'].to(device)
        reshaped_data['answer_mode'] = 'mcq'
    if batch.get('cand_words') is not None:
        reshaped_data['cand_words'] = batch['cand_words'].to(device).float()
    if batch.get('mcq_forward_mode') is not None:
        mode = batch['mcq_forward_mode']
        if isinstance(mode, (list, tuple)):
            mode = mode[0]
        reshaped_data['mcq_forward_mode'] = str(mode)
    if batch.get('quest_words') is not None:
        reshaped_data['quest_words'] = batch['quest_words'].to(device).float()
    if isinstance(reshaped_data['quest'], dict):
        reshaped_data['quest'] = {k: v.to(device) for k, v in reshaped_data['quest'].items()}
    else:
        reshaped_data['quest'] = reshaped_data['quest'].to(device)
    return reshaped_data
