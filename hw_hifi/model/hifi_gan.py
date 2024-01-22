from typing import Union
import itertools
from torch import Tensor, nn
from hw_hifi.base import BaseModel
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from hw_hifi.model.generator import Generator
from hw_hifi.model.mp_discriminator import MPDiscriminator
from hw_hifi.model.ms_discriminator import MSDiscriminator
from hw_hifi.utils.mel import MelSpectrogram
from hw_hifi.loss.discriminator_loss import DiscriminatorLoss
from hw_hifi.loss.feature_loss import FeatureLoss
from hw_hifi.loss.generator_loss import GeneratorLoss

class HiFiGAN(BaseModel):
    def __init__(self, init_channels, kernel_sizes, strides, res_kernels, res_dilations, use_spectral_norm=False, **batch):
        super().__init__(**batch)
        self.generator = Generator(init_channels, kernel_sizes, strides, res_kernels, res_dilations)
        self.mp_discriminator = MPDiscriminator(use_spectral_norm=use_spectral_norm)
        self.ms_discriminator = MSDiscriminator()
        self.mel = MelSpectrogram()
        self.discr_loss = DiscriminatorLoss()
        self.feature_loss = FeatureLoss()
        self.generator_loss = GeneratorLoss()

    def generator_params(self):
        return filter(lambda p: p.requires_grad, self.generator.parameters())
    
    def discriminator_params(self):
        return list(filter(lambda p: p.requires_grad, self.mp_discriminator.parameters())) + \
                list(filter(lambda p: p.requires_grad, self.ms_discriminator.parameters()))
    
    def forward(self, **batch) -> Tensor | dict:
        return batch
    
    def num_params(self):
        print("MP d params:", sum(p.numel() for p in self.mp_discriminator.parameters() if p.requires_grad))
        print("MS d params", sum(p.numel() for p in self.ms_discriminator.parameters() if p.requires_grad))
        print("Generator params:", sum(p.numel() for p in self.generator.parameters() if p.requires_grad))
        print("HiFiGAN params:", sum(p.numel() for p in self.parameters() if p.requires_grad))






        