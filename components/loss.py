import torch
import torch.nn as nn
import torch.nn.functional as F

class MirrorMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='mean')
    
    def forward(self, prediction, target):
        add_vec = torch.torch.tensor(
            [0.5,0,0], dtype=target.dtype, device=target.device)
        mult_vec = torch.tensor(
            [-1,1,1], dtype=target.dtype, device=target.device)
        target_mirror = (target - add_vec) * mult_vec + add_vec

        reg_loss = self.mse(prediction, target)
        mirror_loss = self.mse(prediction, target_mirror)
        loss = torch.min(reg_loss, mirror_loss)

        return loss
