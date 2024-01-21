from typing import Callable

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from src.model.base_discriminator import BaseDiscriminator


class PeriodDiscriminator(nn.Module):
    def __init__(
        self,
        period,
        channels,
        kernel_size=5,
        stride=3,
        norma_fun: Callable = weight_norm,
    ):
        super().__init__()
        self.period = period
        self.layers = nn.ModuleList()

        for i in range(len(channels) - 1):
            layer = [
                norma_fun(
                    nn.Conv2d(
                        channels[i],
                        channels[i + 1],
                        (kernel_size, 1),
                        stride=(1 if i >= len(channels) - 2 else stride, 1),
                        padding=(2, 0),
                    )
                ),
                nn.LeakyReLU(0.1),
            ]
            self.layers.append(nn.Sequential(*layer))
        self.layers.append(
            norma_fun(
                nn.Conv2d(
                    1024,
                    1,
                    (3, 1),
                    1,
                    padding=(1, 0),
                )
            )
        )

    def make2D(self, x):
        B, C, T = x.shape
        if T % self.period != 0:
            n_pad = self.period - (T % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            T = T + n_pad
        return x.view(B, C, T // self.period, self.period)

    def forward(self, x):
        features_list = []
        x = self.make2D(x)
        for layer in self.layers:
            x = layer(x)
            features_list.append(x)
        return x.flatten(1, -1), features_list


class MultiPeriodDiscriminator(BaseDiscriminator):
    def __init__(self, period_list, channels_list, **kwargs):
        super().__init__()
        self.sub_discriminators = nn.ModuleList(
            [PeriodDiscriminator(period, channels_list) for period in period_list]
        )

    def forward(self, **batch):
        (
            mp_outputs_true,
            mp_outputs_fake,
            mp_features_true,
            mp_features_fake,
        ) = super().forward(**batch)
        return {
            "mp_outputs_true": mp_outputs_true,
            "mp_outputs_fake": mp_outputs_fake,
            "mp_features_true": mp_features_true,
            "mp_features_fake": mp_features_fake,
        }
