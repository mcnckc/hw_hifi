import torch
from torch import Tensor
from torch import nn

class DiscriminatorLoss(nn.Module):
    def forward(self, true_outs, fake_outs,
                **batch) -> Tensor:
        loss = 0
        for t_out, f_out in zip(true_outs, fake_outs):
            loss += torch.mean((1 - t_out) ** 2) + torch.mean(f_out ** 2)
        return loss

