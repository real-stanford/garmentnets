import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia


class SpatialGradientMSELoss(nn.Module):
    def __init__(self, mode='diff', order=1):
        super().__init__()
        self.spatial_gradient = kornia.filters.SpatialGradient3d(mode=mode, order=order)
        self.mse = nn.MSELoss(reduction='mean')
    
    def forward(self, prediction, target):
        pred_gradient = self.spatial_gradient(prediction)
        target_gradient = self.spatial_gradient(target)
        loss = self.mse(pred_gradient, target_gradient)
        return loss


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
