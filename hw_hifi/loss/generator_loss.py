import torch
from torch import Tensor
from torch import nn

class GeneratorLoss(nn.Module):
    def forward(self, disc_outs,
                **batch) -> Tensor:
        loss = 0
        for out in disc_outs:
            loss += torch.mean((1 - out) ** 2)
        return loss

