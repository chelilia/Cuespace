"""CueSpace test-only model: InputProj → QMSFE → QMDCR → QCAF."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from src.cuespace.defaults import apply_test_defaults
from src.cuespace.input_proj import InputProj
from src.cuespace.qcaf import QCAF
from src.cuespace.qmdcr import QMDCR
from src.cuespace.qmsfe import QMSFE


class CueSpace(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        cfg = apply_test_defaults(kwargs)
        self.num_labels = int(cfg['num_labels'])
        self.answer_mode = str(cfg['answer_mode'])
        self.mcq_forward_mode = str(cfg.get('mcq_forward_mode', 'shared_stacked'))

        self.input_proj = InputProj(
            d_model=512,
            video_dim=int(cfg['video_dim']),
            patch_dim=int(cfg['patch_dim']),
            audio_dim=int(cfg['audio_dim']),
            audio_patch_dim=int(cfg['audio_patch_dim']),
            encoder_type=str(cfg['encoder_type']),
            use_quest_norm=bool(cfg['use_quest_norm']),
        )
        self.qmsfe = QMSFE(d_model=512, align_dim=int(cfg['align_dim']))
        self.qmdcr = QMDCR()
        self.qcaf = QCAF(num_labels=self.num_labels)

    def _prepare_mcq_per_option_batch(self, reshaped_data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        cand_quest = reshaped_data.get('cand_quest')
        cand_words = reshaped_data.get('cand_words')
        if cand_quest is None:
            raise RuntimeError('per_option_full requires cand_quest')
        b, k = cand_quest.shape[:2]
        if k != 4:
            raise ValueError(f'per_option_full expects 4 options, got K={k}')

        def _expand(t: Optional[Tensor]) -> Optional[Tensor]:
            if t is None:
                return None
            return t.repeat_interleave(k, dim=0)

        out = dict(reshaped_data)
        out['audio'] = _expand(out['audio'])
        out['video'] = _expand(out['video'])
        out['patch'] = _expand(out.get('patch'))
        out['audio_patch'] = _expand(out.get('audio_patch'))
        out['quest'] = cand_quest.reshape(b * k, -1)
        if cand_words is not None:
            out['quest_words'] = cand_words.reshape(b * k, cand_words.shape[2], -1)
        else:
            out.pop('quest_words', None)
        out['_mcq_original_batch'] = b
        out['mcq_forward_mode'] = 'per_option_full'
        return out

    def forward(self, reshaped_data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        mcq_mode = reshaped_data.get('mcq_forward_mode', self.mcq_forward_mode)
        if reshaped_data.get('answer_mode') == 'mcq' and mcq_mode == 'per_option_full':
            reshaped_data = self._prepare_mcq_per_option_batch(reshaped_data)

        quest_raw = reshaped_data['quest']
        audio = reshaped_data['audio']
        video = reshaped_data['video']
        patch = reshaped_data.get('patch')
        audio_patch = reshaped_data.get('audio_patch')
        quest_words_offline = reshaped_data.get('quest_words')

        quest, words = self.input_proj.encode_text(quest_raw, quest_words_offline)
        quest_proj, words_proj, audio, video, patch, audio_patch = self.input_proj.project_modalities(
            audio=audio,
            video=video,
            patch=patch,
            audio_patch=audio_patch,
            words=words,
            quest=quest,
        )

        audio, video, audio_shared, video_shared = self.qmsfe.enhance(
            audio, video, words_proj,
        )

        a_seq, v_seq, s_seq = self.qmdcr(
            quest_proj=quest_proj,
            audio=audio,
            video=video,
            audio_shared=audio_shared,
            video_shared=video_shared,
            patch=patch,
            audio_patch=audio_patch,
        )

        answer_mode = str(reshaped_data.get('answer_mode', self.answer_mode))
        cand_proj = None
        if answer_mode == 'mcq' and mcq_mode != 'per_option_full':
            cand_quest = reshaped_data.get('cand_quest')
            if cand_quest is None:
                raise RuntimeError('answer_mode=mcq requires cand_quest')
            cand_proj = self.input_proj.project_candidate_quests(cand_quest)

        cm_out = self.qcaf(
            a_seq, v_seq, s_seq,
            quest=quest_proj,
            words=words_proj,
            answer_mode=answer_mode,
            mcq_forward_mode=str(reshaped_data.get('mcq_forward_mode', mcq_mode)),
            cand_quest_proj=cand_proj,
            original_batch=reshaped_data.get('_mcq_original_batch'),
        )
        return {'out': cm_out['out']}
