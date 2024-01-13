from typing import List

import torch
from torch import Tensor
import multiprocessing
from hw_hifi.base.base_metric import BaseMetric
from hw_hifi.base.base_text_encoder import BaseTextEncoder
from hw_hifi.metric.utils import calc_wer
from pyctcdecode import build_ctcdecoder
from hw_hifi.utils import ROOT_PATH

class PyCTCBeamSearchWERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, model_path, beam_size=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = beam_size
        self.decoder = build_ctcdecoder(
            [text_encoder.EMPTY_TOK] + list(text_encoder.alphabet),
            kenlm_model_path=model_path,  # either .arpa or .bin file
            alpha=0.5,  # tuned on a val set
            beta=1.0,  # tuned on a val set
        )

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        cers = []
        lengths = log_probs_length.detach().numpy()
        with multiprocessing.get_context("fork").Pool() as pool:
            pred_list = self.decoder.decode_batch(pool, log_probs.detach(), beam_width=self.beam_size)
        for pred_text, target_text in zip(pred_list, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            cers.append(calc_wer(target_text, pred_text))
        return sum(cers) / len(cers)
