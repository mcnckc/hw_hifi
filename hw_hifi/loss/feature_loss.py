import torch
from torch import Tensor
from torch import nn

class FeatureLoss(nn.Module):
    def forward(self, true_fs, fake_fs,
                **batch) -> Tensor:
        loss = 0
        for t_fs_list, f_fs_list in zip(true_fs, fake_fs):
            for t_fs, f_fs in zip(t_fs_list, f_fs_list):
                print("F shapes:", t_fs.shape, f_fs.shape)
                loss += torch.mean(torch.abs(t_fs - f_fs))    
        return loss ** 2

