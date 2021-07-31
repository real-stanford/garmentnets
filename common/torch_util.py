import torch
import numpy as np
import torch_geometric.data as tgd

def to_numpy(x):
    return x.detach().to('cpu').numpy()

def get_batch_size(obj):
    if isinstance(obj, torch.Tensor):
        return obj.shape[0]
    elif isinstance(obj, tgd.Batch):
        return obj.num_graphs
    else:
        raise TypeError("Unsupported Type")
