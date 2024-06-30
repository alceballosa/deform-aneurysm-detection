import math

import numpy as np
import torch


def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)



def get_3d_sinusoidal_pos_emb(pos, num_pos_feats=128, temperature=10000):
    # https://github.com/megvii-research/PETR/blob/main/projects/mmdet3d_plugin/models/dense_heads/petr_head.py#L29
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack(
        (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    pos_y = torch.stack(
        (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    pos_z = torch.stack(
        (pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    return posemb


def get_3d_corners(center, size):
    """
    Given a list of centers and offsets, return the 8 corners of the bounding boxes as a
    list shaped (B, 8, 3).

    Args:
        center: (B, 3), the center of the boxes.
        offset: (B, 3), the sizes of the boxes.

    Returns:
        corners: (B, 8, 3), the corners of the boxes.
    """
    x_min = center[:, 0] - size[:, 0] / 2
    x_max = center[:, 0] + size[:, 0] / 2
    y_min = center[:, 1] - size[:, 1] / 2
    y_max = center[:, 1] + size[:, 1] / 2
    z_min = center[:, 2] - size[:, 2] / 2
    z_max = center[:, 2] + size[:, 2] / 2

    corners = torch.stack(
        [
            x_min,
            y_min,
            z_min,
            x_max,
            y_min,
            z_min,
            x_max,
            y_max,
            z_min,
            x_min,
            y_max,
            z_min,
            x_min,
            y_min,
            z_max,
            x_max,
            y_min,
            z_max,
            x_max,
            y_max,
            z_max,
            x_min,
            y_max,
            z_max,
        ],
        dim=-1,
    )
    corners = corners.reshape(*corners.shape[:-1], 8, 3)
    return corners
