import logging
import torch
import torch.nn as nn
from typing import List
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    result_batch = {}
    """
    result_batch['text_encoded_length'] = torch.tensor([item['text_encoded'].shape[-1] for item in dataset_items])
    result_batch['text'] = [item['text'] for item in dataset_items]
    max_len = result_batch['text_encoded_length'].max()
    result_batch['text_encoded'] = torch.cat([nn.ConstantPad1d((0, max_len - item['text_encoded'].shape[-1]), 0)
                                              (item['text_encoded']) for item in dataset_items])
    """

    """
    result_batch['spectrogram_length'] = torch.tensor([item['spectrogram'].shape[-1] for item in dataset_items])
    max_sp_len = result_batch['spectrogram_length'].max()
    spectral_silence = torch.log(torch.tensor(1e-5)).item()
    result_batch['spectrogram'] = torch.stack([nn.ConstantPad1d((0, max_sp_len - item['spectrogram'].shape[-1]), spectral_silence)
                                               (item['spectrogram'].squeeze()) for item in dataset_items])
    result_batch['audio_length'] = torch.tensor([item['audio'].shape[-1] for item in dataset_items])
    """
    result_batch['audio_wave'] = pad_sequence([item['audio'].squeeze() for item in dataset_items], batch_first=True, padding_value=0)
    return result_batch