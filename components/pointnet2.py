import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import PointConv, fps, radius, global_max_pool, knn_interpolate
from torch.cuda.amp import autocast


class SAModule(torch.nn.Module):
    """
    Local Set Abstraction (convolution)
    """
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        # pointnet1 module
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch):
        # furtherest point sampling
        edge_index = None
        with autocast(enabled=False):
            idx = fps(pos, batch, ratio=self.ratio)
            # ball query
            row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                            max_num_neighbors=64)
            edge_index = torch.stack([col, row], dim=0)
            x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    """
    Global Set Abstraction
    """
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        with autocast(enabled=False):
            # TODO
            x = x.type(torch.float32)
            x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])

class FPModule(torch.nn.Module):
    """
    Feature Propogation (deconvolution)
    """
    def __init__(self, k, nn):
        super(FPModule, self).__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        with autocast(enabled=False):
            x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
            if x_skip is not None:
                x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip
