from dataclasses import dataclass
import torch
from torch import nn

import torchaudio

import librosa
from torchaudio.transforms import InverseMelScale


@dataclass
class MelSpectrogramConfig:
    sr: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    f_min: int = 0
    f_max: int = 8000
    n_mels: int = 80
    power: float = 1.0
    pad_value: float = -11.5129251


class MelSpectrogram(nn.Module):
    def __init__(
        self,
        config: MelSpectrogramConfig,
    ):
        super(MelSpectrogram, self).__init__()

        self.config = config

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sr,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_fft=config.n_fft,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
            center=False,
            power=config.power,
            pad=(config.win_length - config.hop_length) // 2,
        )

        # The ииis no way to set power in constructor in 0.5.0 version.
        # self.mel_spectrogram.spectrogram.power = config.power

        # Default `torchaudio` mel_true basis uses HTK formula. In order to be compatible with WaveGlow
        # we decided to use Slaney one instead (as well as `librosa` does by default).
        mel_basis = librosa.filters.mel(
            sr=config.sr,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=config.f_min,
            fmax=config.f_max,
        ).T
        self.mel_spectrogram.mel_scale.fb.copy_(torch.tensor(mel_basis))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        :param audio: Expected shape is [B, T]
        :return: Shape is [B, n_mels, T']
        """
        return self.mel_spectrogram(audio).clamp_(min=1e-5).log_()


class GriffinLim(torchaudio.transforms.GriffinLim):
    """GriffinLim algorithm as baseline for our task"""

    def __init__(self):
        self.config = MelSpectrogramConfig()
        super().__init__(
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            power=self.config.power,
        )
        self.transform = InverseMelScale(
            sample_rate=self.config.sr,
            n_stft=self.config.n_fft // 2 + 1,
            f_min=self.config.f_min,
            f_max=self.config.f_max,
            norm="slaney",
            mel_scale="slaney",
            n_mels=self.config.n_mels,
        )

    def forward(self, mel_spec):
        return super().forward(self.transform(mel_spec.exp()))
