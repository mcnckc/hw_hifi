import torch
from torch import Tensor
from torch import nn

class GeneratorLoss(nn.Module):
    def forward(self, disc_outs,
                **batch) -> Tensor:
        print("Calculate generator loss")
        loss = 0
        for out in disc_outs:
            loss += torch.mean((1 - out) ** 2)
            print("DISCR CUR LOSS", torch.max(out), torch.min(out), torch.median(out), torch.mean(out))
        return loss

