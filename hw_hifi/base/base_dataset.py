import logging
import random
from typing import List

import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

from hw_hifi.utils.parse_config import ConfigParser
from hw_hifi.utils.mel import MelSpectrogram

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
            self,
            index,
            config_parser: ConfigParser,
            wave_augs=None,
            spec_augs=None,
            limit=None,
            segment_length=None,
            max_audio_length=None,
            max_text_length=None,
    ):
        self.config_parser = config_parser
        self.wave_augs = wave_augs
        self.spec_augs = spec_augs
        self.segment_length = segment_length
        #self.log_spec = config_parser["preprocessing"]["log_spec"]

        self._assert_index_is_valid(index)
        index = self._filter_records_from_dataset(index, max_audio_length, max_text_length, limit)
        # it's a good idea to sort index by audio length
        # It would be easier to write length-based batch samplers later
        index = self._sort_index(index)
        self._index: List[dict] = index

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        audio_path = data_dict["path"]
        audio_wave = self.load_audio(audio_path)
        if audio_wave.shape[-1] > self.segment_length:
            start = np.random.randint(audio_wave.shape[-1] - self.segment_length + 1)
            #start = 0
            segment = audio_wave[..., start:start + self.segment_length]
        else:
            segment = audio_wave
        segment, audio_spec = self.process_wave(segment)
        return {
            "audio": segment,
            "spectrogram": audio_spec,
            "duration": segment.size(1) / self.config_parser["preprocessing"]["sr"],
            "audio_path": audio_path,
        }

    @staticmethod
    def _sort_index(index):
        return sorted(index, key=lambda x: x["audio_len"])

    def __len__(self):
        return len(self._index)

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.config_parser["preprocessing"]["sr"]
        if sr != target_sr:
            print("Sample rate mismatch!")
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    def process_wave(self, audio_tensor_wave: Tensor):
        with torch.no_grad():
            if self.wave_augs is not None:
                audio_tensor_wave = self.wave_augs(audio_tensor_wave)
            wave2spec = MelSpectrogram()
            audio_tensor_spec = wave2spec(audio_tensor_wave)
            #if self.spec_augs is not None:
                #audio_tensor_spec = self.spec_augs(audio_tensor_spec)
            #if self.log_spec:
                #audio_tensor_spec = torch.log(audio_tensor_spec + 1e-5)
            return audio_tensor_wave, audio_tensor_spec

    @staticmethod
    def _filter_records_from_dataset(
            index: list, max_audio_length, max_text_length, limit
    ) -> list:
        initial_size = len(index)
        if max_audio_length is not None:
            exceeds_audio_length = np.array([el["audio_len"] for el in index]) >= max_audio_length
            _total = exceeds_audio_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer then "
                f"{max_audio_length} seconds. Excluding them."
            )
        else:
            exceeds_audio_length = False

        initial_size = len(index)
        exceeds_text_length = False

        records_to_filter = exceeds_text_length | exceeds_audio_length

        if records_to_filter is not False and records_to_filter.any():
            _total = records_to_filter.sum()
            index = [el for el, exclude in zip(index, records_to_filter) if not exclude]
            logger.info(
                f"Filtered {_total}({_total / initial_size:.1%}) records  from dataset"
            )

        if limit is not None:
            random.seed(42)  # best seed for deep learning
            random.shuffle(index)
            index = index[:limit]
        return index

    @staticmethod
    def _assert_index_is_valid(index):
        for entry in index:
            assert "audio_len" in entry, (
                "Each dataset item should include field 'audio_len'"
                " - duration of audio (in seconds)."
            )
            assert "path" in entry, (
                "Each dataset item should include field 'path'" " - path to audio file."
            )
            assert "text" in entry, (
                "Each dataset item should include field 'text'"
                " - text transcription of the audio."
            )
