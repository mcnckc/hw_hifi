import torch
import torch.nn as nn

class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _loss_fn(
        disc_outputs_real,
        disc_outputs_generated,
    ) -> torch.Tensor:
        return sum(
            (1.0 - dr).square().mean() + dg.square().mean()
            for dr, dg in zip(
                disc_outputs_real,
                disc_outputs_generated,
            )
        )

    def forward(
        self,
        mp_outputs_true,
        mp_outputs_fake,
        ms_outputs_true,
        ms_outputs_fake,
        **kwargs
    ):
        loss_disc_f = self._loss_fn(mp_outputs_true, mp_outputs_fake)
        loss_disc_s = self._loss_fn(ms_outputs_true, ms_outputs_fake)
        return {"discriminator_loss": loss_disc_s + loss_disc_f}


class GeneratorLoss(nn.Module):
    def __init__(self, generator_loss_scale, mel_loss_scale, feature_loss_scale):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.generator_loss_scale = generator_loss_scale
        self.mel_loss_scale = mel_loss_scale
        self.feature_loss_scale = feature_loss_scale

    def _feature_loss_fn(self, true_list, fake_list) -> torch.Tensor:
        return sum(
            self.l1_loss(true, fake) * self.feature_loss_scale
            for true_sublist, fake_sublist in zip(true_list, fake_list)
            for true, fake in zip(true_sublist, fake_sublist)
        )

    def _generator_loss_fn(self, discriminator_outputs):
        return sum(
            (1.0 - dg).square().mean() * self.generator_loss_scale
            for dg in discriminator_outputs
        )

    def forward(
        self,
        mel_true,
        mel_fake,
        mp_features_true,
        mp_features_fake,
        ms_features_true,
        ms_features_fake,
        mp_outputs_fake,
        ms_outputs_fake,
        **kwargs
    ):
        mp_feature_loss = self._feature_loss_fn(mp_features_true, mp_features_fake)
        ms_feature_loss = self._feature_loss_fn(ms_features_true, ms_features_fake)
        mp_generator_loss = self._generator_loss_fn(mp_outputs_fake)
        ms_generator_loss = self._generator_loss_fn(ms_outputs_fake)
        loss_mel = self.l1_loss(mel_true, mel_fake)
        return {
            "generator_loss": ms_generator_loss
            + mp_generator_loss
            + ms_feature_loss
            + mp_feature_loss
            + loss_mel * self.mel_loss_scale,
            "mel_loss": loss_mel.detach(),
            "feature_loss": (mp_feature_loss.detach() + ms_feature_loss.detach())
            / self.feature_loss_scale,
            "generator_gan_loss": (
                ms_generator_loss.detach() + mp_generator_loss.detach()
            )
            / self.generator_loss_scale,
        }


class HiFiGANLoss(nn.Module):
    def __init__(self, generator_loss_scale, mel_loss_scale, feature_loss_scale):
        super().__init__()
        self.generator_loss = GeneratorLoss(
            generator_loss_scale, mel_loss_scale, feature_loss_scale
        )
        self.discriminator_loss = DiscriminatorLoss()