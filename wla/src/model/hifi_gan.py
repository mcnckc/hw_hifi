from torch import nn
from src.model.generator import Generator
from src.model.mp_discriminator import MultiPeriodDiscriminator
from src.model.ms_discriminator import MultiScaleDiscriminator


class HiFiGAN(nn.Module):
    """HiFiGAN"""

    def __init__(
        self,
        res_kernel_sizes,
        res_dilation_sizes,
        up_init_channels,
        up_strides,
        up_kernels,
        mp_period_list,
        mp_channels_list,
        ms_channels_list,
        ms_kernels_list,
        ms_strides_list,
        ms_groups_list,
        **kwargs
    ):
        super().__init__()

        self.generator = Generator(
            res_kernel_sizes=res_kernel_sizes,
            res_dilation_sizes=res_dilation_sizes,
            up_init_channels=up_init_channels,
            up_strides=up_strides,
            up_kernels=up_kernels,
        )
        self.mp_discriminator = MultiPeriodDiscriminator(
            period_list=mp_period_list, channels_list=mp_channels_list
        )
        self.ms_discriminator = MultiScaleDiscriminator(
            ms_channels_list=ms_channels_list,
            ms_kernels_list=ms_kernels_list,
            ms_strides_list=ms_strides_list,
            ms_groups_list=ms_groups_list,
        )

    def forward(self, waves, **kwargs):
        return {
            "mel_true": self.generator.mel(waves),
            "wave_true": waves.unsqueeze(dim=1),
        }
