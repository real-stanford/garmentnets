from typing import Optional
import torch


def mirror_nocs_points_by_axis(points: torch.Tensor, axis: Optional[int]=None):
    if axis is None:
        return points

    add_vec_list = [0,0,0]
    mul_vec_list = [1,1,1]
    add_vec_list[axis] = 0.5
    mul_vec_list[axis] = -1

    add_vec = torch.torch.tensor(
        add_vec_list, dtype=points.dtype, device=points.device)
    mult_vec = torch.tensor(
        mul_vec_list, dtype=points.dtype, device=points.device)
    points_mirror = (points - add_vec) * mult_vec + add_vec
    return points_mirror
