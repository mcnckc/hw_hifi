import os
import random

import librosa
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from src.model.mel_spectrogram import (
    MelSpectrogram,
    MelSpectrogramConfig,
)


class BufferDataset(Dataset):
    def __init__(self, wav_dir, max_len=22528, sr=22050, **kwargs):
        self.wav_dir = wav_dir
        self.max_len = max_len
        mel_config = MelSpectrogramConfig()
        self.mel_creator = MelSpectrogram(mel_config)
        self.file_list = [
            os.path.join(wav_dir, filename)
            for filename in os.listdir(wav_dir)
            if filename.endswith(".wav")
        ]
        self.buffer = []
        for file_path in tqdm(iterable=sorted(self.file_list), desc="Loading dataset"):
            waveform, _ = librosa.load(file_path, sr=sr)
            entry = {
                "wave": waveform,
                "path": file_path,
            }
            self.buffer.append(entry)
        self.length_dataset = len(self.buffer)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        entry = self.buffer[idx]
        waveform = entry["wave"]
        if len(waveform) >= self.max_len:
            max_start_index = len(waveform) - self.max_len
            start_index = random.randint(0, max_start_index)
            waveform_segment = waveform[start_index : start_index + self.max_len]
        else:
            waveform_segment = waveform
        return {
            "wave": waveform_segment,
            "path": entry["path"],
        }
