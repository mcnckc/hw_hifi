import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from src.model.mel_spectrogram import (
    MelSpectrogram,
    MelSpectrogramConfig,
)
from src.model.utils import init_generator_weights


class ResBlock(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=3,
        dilations=(1, 3, 5),
    ):
        super(ResBlock, self).__init__()
        self.blocks = nn.ModuleList()

        for dilation in dilations:
            conv_layers = nn.Sequential(
                nn.LeakyReLU(0.1),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        dilation=dilation,
                        padding="same",
                    )
                ),
                nn.LeakyReLU(0.1),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        padding="same",
                    )
                ),
            )
            conv_layers.apply(init_generator_weights)
            self.blocks.append(conv_layers)

    def forward(self, x):
        for conv in self.blocks:
            x = x + conv(x)
        return x


class MRF(nn.Module):
    def __init__(
        self,
        in_channels,
        res_kernel_list,
        res_dilation_list,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [
                ResBlock(
                    in_channels // 2,
                    kernel,
                    dilation,
                )
                for kernel, dilation in zip(
                    res_kernel_list,
                    res_dilation_list,
                )
            ]
        )

    def forward(self, x):
        return sum(block(x) for block in self.res_blocks) / len(self.res_blocks)


class Generator(torch.nn.Module):
    def __init__(
        self,
        res_kernel_sizes,
        res_dilation_sizes,
        up_init_channels,
        up_strides,
        up_kernels,
    ):
        super().__init__()
        blocks = [weight_norm(nn.Conv1d(80, up_init_channels, 7, 1, 3))]
        channels = up_init_channels
        for stride, kernel_size in zip(up_strides, up_kernels):
            pad = (kernel_size - stride) // 2
            blocks.extend(
                [
                    nn.LeakyReLU(0.1),
                    weight_norm(
                        nn.ConvTranspose1d(
                            channels,
                            channels // 2,
                            kernel_size,
                            stride=stride,
                            padding=pad,
                        )
                    ),
                    MRF(
                        channels,
                        res_kernel_sizes,
                        res_dilation_sizes,
                    ),
                ]
            )
            channels //= 2

        blocks.extend(
            [
                nn.LeakyReLU(0.1),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        out_channels=1,
                        kernel_size=7,
                        padding="same",
                    )
                ),
                nn.Tanh(),
            ]
        )
        self.net = nn.Sequential(*blocks)
        self.net.apply(init_generator_weights)
        self.mel = MelSpectrogram(MelSpectrogramConfig())

    def forward(self, mel_true, **batch):
        wave_fake = self.net(mel_true.squeeze(1))
        return {
            "wave_fake": wave_fake,
            "wave_fake_detached": wave_fake.detach(),
            "mel_fake": self.mel(wave_fake.squeeze(1)),
        }
