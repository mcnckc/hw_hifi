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
    
class ShallowSpeechConv(BaseModel):
    def __init__(self, n_feats, n_class, n_hidden=512, n_channels=8, n_rlayers=2, **batch):
        super().__init__(n_feats, n_class, **batch)
        self.n_hidden = n_hidden
        self.n_channels = n_channels
        self.kernels = [(21, 6), (11, 6)]
        self.strides = [(2, 2), (2, 1)]
        self.conv = Sequential(
            nn.Conv2d(1, n_channels, self.kernels[0], stride=self.strides[0]),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels, self.kernels[1], stride=self.strides[1]),
            nn.BatchNorm2d(n_channels),
            nn.ReLU()
        )
        self.recurrent = nn.GRU(self._transform_features_length(n_feats), n_hidden, batch_first=True, bidirectional=True, num_layers=n_rlayers)
        self.bn = nn.BatchNorm1d(n_hidden * 2)
        self.final = Sequential(
            nn.ReLU(),
            FF(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_class)
        )
    def _transform_features_length(self, n_feats):
        n = n_feats
        for i in range(2):
            n = (n - self.kernels[i][0]) // self.strides[i][0] + 1
        return n * self.n_channels
    
    def forward(self, spectrogram, **batch):
        x = self.conv(spectrogram.unsqueeze(dim=1))
        x, _ = self.recurrent(x.flatten(start_dim=1, end_dim=2).transpose(1, 2))
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        return {"logits": self.final(x[:, :, :self.n_hidden] + x[:, :, self.n_hidden:])}

    def transform_input_lengths(self, input_lengths):
        for i in range(2):
            input_lengths = (input_lengths - self.kernels[i][1]) // self.strides[i][1] + 1
        return input_lengths
