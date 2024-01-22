from typing import Union
import itertools
from torch import Tensor, nn
from hw_hifi.base import BaseModel
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from hw_hifi.model.wla_generator import WLAGenerator
from hw_hifi.model.wla_mp_discirminator import WLAMultiPeriodDiscriminator
from hw_hifi.model.wla_ms_discriminator import WLAMultiScaleDiscriminator
from hw_hifi.utils.mel import MelSpectrogram
from hw_hifi.loss.discriminator_loss import DiscriminatorLoss
from hw_hifi.loss.feature_loss import FeatureLoss
from hw_hifi.loss.generator_loss import GeneratorLoss

import torch.nn.functional as F

class GeneratorConfig:
    def __init__(self, init_channels, kernel_sizes, strides, res_kernels, res_dilations):
        self.upsample_rates = strides
        self.upsample_kernel_sizes = kernel_sizes
        self.upsample_initial_channel = init_channels
        self.resblock_kernel_sizes = res_kernels
        self.resblock_dilation_sizes = res_dilations
        self.resblock = '1'

class WLAHiFiGAN(BaseModel):
    def __init__(self, init_channels, kernel_sizes, strides, res_kernels, res_dilations, **batch):
        super().__init__(**batch)
        self.generator = WLAGenerator(res_kernels, res_dilations, init_channels, strides, kernel_sizes)
        self.mp_discriminator = WLAMultiPeriodDiscriminator([2, 3, 5, 7, 11], [1, 32, 128, 512, 1024, 1024])
        self.ms_discriminator = WLAMultiScaleDiscriminator()
        self.mel = MelSpectrogram()
        self.discr_loss = DiscriminatorLoss()
        self.feature_loss = FeatureLoss()
        self.generator_loss = GeneratorLoss()
        print("USING WLA MODEL")

    def generator_params(self):
        return filter(lambda p: p.requires_grad, self.generator.parameters())
    
    def discriminator_params(self):
        return list(filter(lambda p: p.requires_grad, self.mp_discriminator.parameters())) + \
                list(filter(lambda p: p.requires_grad, self.ms_discriminator.parameters()))
    def num_params(self):
        print("MP d params:", sum(p.numel() for p in self.mp_discriminator.parameters() if p.requires_grad))
        print("MS d params", sum(p.numel() for p in self.ms_discriminator.parameters() if p.requires_grad))
        print("Generator params:", sum(p.numel() for p in self.generator.parameters() if p.requires_grad))
        print("HiFiGAN params:", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, **batch) -> Tensor | dict:
        return batch






        