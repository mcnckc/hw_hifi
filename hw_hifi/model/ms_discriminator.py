from torch import nn
from hw_hifi.base import BaseModel
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
import torch.nn.functional as F

class SDiscriminator(nn.Module):
    def __init__(self, use_spectral_norm=False, *args, **kwargs, ) -> None:
        super().__init__(*args, **kwargs)
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        [128, 128, 256, 512, 1024, 1024, 1024]
        channels = [1, 128, 128, 256, 512, 1024, 1024, 1024, 1]
        kss = [15] + [41] * 5 + [5, 3]
        strides = [1, 2, 2, 4, 4, 1, 1, 1]
        groups = [1, 4] + [16] * 4 + [1, 1]
        padding = [7] + [20] * 5 + [2, 1]
        n_layers = len(channels) - 1
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i < n_layers - 1:
                self.layers.append(nn.Sequential(
                    norm_f(nn.Conv1d(channels[i], channels[i + 1], kss[i], strides[i], groups=groups[i], padding=padding[i])),
                    nn.LeakyReLU(0.1)
                ))
            else:
                self.layers.append(nn.Sequential(
                    norm_f(nn.Conv1d(channels[i], channels[i + 1], kss[i], strides[i], groups=groups[i], padding='same'))
                ))
    def forward(self, x):
        features = []
        for l in self.layers:
            x = l(x)
            features.append(x)
        return x.flatten(1, -1), features
    
class MSDiscriminator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.discriminators = nn.ModuleList([
            SDiscriminator(use_spectral_norm=True),
            nn.Sequential(nn.AvgPool1d(4, 2, padding=2), SDiscriminator()),
            nn.Sequential(nn.AvgPool1d(4, 2, padding=2), SDiscriminator())
        ])
    def forward(self, true, fake):
        true_out, fake_out, true_fs, fake_fs = [], [], [], []
        for d in self.discriminators:
            out, fs = d(true)
            true_out.append(out)
            true_fs.append(fs)
            out, fs = d(fake)
            fake_out.append(out)
            fake_fs.append(fs)
        return true_out, fake_out, true_fs, fake_fs

