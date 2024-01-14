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

import torch.nn.functional as F

class HiFiGAN(BaseModel):
    def __init__(self, init_channels, kernel_sizes, strides, res_kernels, res_dilations, **batch):
        super().__init__(**batch)
        self.generator = Generator(init_channels, kernel_sizes, strides, res_kernels, res_dilations)
        self.mp_discriminator = MPDiscriminator()
        self.ms_discriminator = MSDiscriminator()
        self.mel = MelSpectrogram()
        self.discr_loss = DiscriminatorLoss()
        self.feature_loss = FeatureLoss()
        self.generator_loss = GeneratorLoss()

    def generator_params(self):
        return self.generator.parameters()
    
    def discriminator_params(self):
        return itertools.chain(self.mp_discriminator.parameters(), self.ms_discriminator.parameters())
    
    def forward(self, **batch) -> Tensor | dict:
        true = batch['audio_wave'].unsqueeze(dim=1)
        raw_fake = self.generator(batch['spectrogram'])
        if raw_fake.shape[-1] >= true.shape[-1]:
            fake = raw_fake[..., :true.shape[-1]].clone()
        else:
            print("Prediction is shorter")
        
        print("FAKE SHAPE:", fake.shape, "TRUE SHAPE:", true.shape)
        fake_spec = self.mel(fake)
        true_out_mp, fake_out_mp, _, _ = self.mp_discriminator(true, fake.detach())
        true_out_ms, fake_out_ms, _, _ = self.ms_discriminator(true, fake.detach())

        d_loss = self.discr_loss(true_out_mp, fake_out_mp) + self.discr_loss(true_out_ms, fake_out_ms)

        loss_mel = F.l1_loss(batch['spectrogram'], fake_spec) * 45

        true_out_mp, fake_out_mp, true_fs_mp, fake_fs_mp = self.mp_discriminator(true, fake)
        true_out_ms, fake_out_ms, true_fs_ms, fake_fs_ms = self.ms_discriminator(true, fake)

        loss_feature = self.feature_loss(true_fs_mp, fake_fs_mp) + self.feature_loss(true_fs_ms, fake_fs_ms)
        loss_gen = self.generator_loss(fake_out_mp) + self.generator_loss(fake_out_ms)
        total_gen_loss = loss_mel + loss_feature + loss_gen

        return {'audio_wave': fake, 
                'd_loss': d_loss, 
                'mel_loss': loss_mel,
                'feature_loss': loss_feature,
                'gen_loss': loss_gen,
                'total_gen_loss': total_gen_loss}






        