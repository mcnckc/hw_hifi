from torch import nn
from hw_hifi.base import BaseModel
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
import torch.nn.functional as F

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

class PDiscriminator(nn.Module):
    def __init__(self, p, ks=5, stride=3, use_spectral_norm=False):
        super().__init__()
        self.p = p
        norm_f = spectral_norm if use_spectral_norm == True else weight_norm
        self.layers = nn.ModuleList()
        channels = [1, 32, 128, 512, 1024, 1024]
        n_layers = len(channels) - 1
        for i in range(n_layers):
            self.layers.append(nn.Sequential(
                norm_f(nn.Conv2d(channels[i], channels[i + 1], (ks, 1),
                                  (stride, 1), padding=(get_padding(ks, 1), 0))),
                nn.LeakyReLU(0.1)
            ))
        self.layers.append(norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0))))

    def reshape_audio(self, x):
        B, C, T = x.shape
        if T % self.period != 0:
            n_pad = self.period - (T % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            T = T + n_pad
        return x.view(B, C, T // self.period, self.period)
    
    def forward(self, x):
        x = self.reshape_audio(x)
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return x.flatten(1, -1), features

class MPDiscriminator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.discriminators = nn.ModuleList([
            PDiscriminator(p) for p in [2, 3, 5, 7, 11]
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


