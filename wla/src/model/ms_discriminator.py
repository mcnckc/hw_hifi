import torch
from typing import Callable
import torch.nn as nn
from torch.nn import AvgPool1d
from torch.nn.utils import weight_norm, spectral_norm
from src.model.base_discriminator import BaseDiscriminator


class ScaleDiscriminator(torch.nn.Module):
    def __init__(
        self,
        norm_fun: Callable,
        ms_channels_list: list[int],
        ms_kernels_list: list[int],
        ms_strides_list: list[int],
        ms_groups_list: list[int],
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        in_channels = 1

        for out_channels, kernel_size, stride, groups in zip(
            ms_channels_list, ms_kernels_list, ms_strides_list, ms_groups_list
        ):
            self.layers.append(
                nn.Sequential(
                    norm_fun(
                        nn.Conv1d(
                            in_channels,
                            out_channels,
                            kernel_size,
                            stride=stride,
                            groups=groups,
                            padding=(kernel_size // 2),
                        )
                    ),
                    nn.LeakyReLU(0.1),
                )
            )
            in_channels = out_channels
        self.layers.append(
            norm_fun(
                nn.Conv1d(
                    in_channels,
                    1,
                    3,
                    1,
                    padding="same",
                )
            )
        )

    def forward(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return x, features


class MultiScaleDiscriminator(BaseDiscriminator):
    def __init__(self, **kwargs):
        super().__init__()
        self.sub_discriminators = nn.ModuleList(
            [
                ScaleDiscriminator(spectral_norm, **kwargs),
                ScaleDiscriminator(weight_norm, **kwargs),
                ScaleDiscriminator(weight_norm, **kwargs),
            ]
        )
        self.pooling = AvgPool1d(4, 2, padding=2)

    def forward(self, **batch):
        (
            ms_outputs_true,
            ms_outputs_fake,
            ms_features_true,
            ms_features_fake,
        ) = super().forward(**batch)
        return {
            "ms_outputs_true": ms_outputs_true,
            "ms_outputs_fake": ms_outputs_fake,
            "ms_features_true": ms_features_true,
            "ms_features_fake": ms_features_fake,
        }
