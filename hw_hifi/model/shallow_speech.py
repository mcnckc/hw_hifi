import torch
from torch import nn
from torch.nn import Sequential

from hw_hifi.base import BaseModel

class FF(nn.Module):
    def __init__(self, n_feats, n_hidden, dropout=0.05):
        super().__init__()
        self.net = Sequential(
            nn.Hardtanh(0, 20),
            nn.Linear(n_feats, n_hidden, bias=True),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    
class ShallowSpeech(BaseModel):
    def __init__(self, n_feats, n_class, n_hidden=512, **batch):
        super().__init__(n_feats, n_class, **batch)
        self.n_hidden = n_hidden
        self.conv = nn.Conv2d(1, 1, 5, padding='same', padding_mode='replicate')
        self.net = Sequential(
            FF(n_feats, n_hidden),
            nn.RNN(n_hidden, n_hidden, batch_first=True, bidirectional=True)
        )
        self.final = Sequential(
            FF(n_hidden, n_hidden),
            nn.Linear(n_hidden, n_class)
        )

    def forward(self, spectrogram, **batch):
        x, _ = self.net(torch.squeeze(self.conv(spectrogram.transpose(1, 2).unsqueeze(dim=1)), dim=1))
        return {"logits": self.final(x[:, :, :self.n_hidden] + x[:, :, self.n_hidden:])}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
