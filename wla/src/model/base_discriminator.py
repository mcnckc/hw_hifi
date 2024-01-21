import torch.nn as nn


class BaseDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.sub_discriminators = nn.ModuleList()
        self.pooling = nn.Identity()

    def forward(self, wave_true, wave_fake_detached, wave_fake, detach=True, **kwargs):
        wave_fake_consistent = wave_fake_detached if detach else wave_fake
        true_outputs, fake_outputs, true_features, fake_features = [], [], [], []

        for sub_discriminator in self.sub_discriminators:
            true_output, true_feature_map = sub_discriminator(wave_true)
            fake_output, fake_feature_map = sub_discriminator(wave_fake_consistent)
            true_outputs.append(true_output)
            true_features.append(true_feature_map)
            fake_outputs.append(fake_output)
            fake_features.append(fake_feature_map)

            wave_true = self.pooling(wave_true)
            wave_fake_consistent = self.pooling(wave_fake_consistent)

        return true_outputs, fake_outputs, true_features, fake_features
