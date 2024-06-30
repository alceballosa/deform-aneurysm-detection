import math

import numpy as np
import torch


def get_3d_sinusoidal_pos_emb(
    pos, num_pos_feats=256, temperature=10000, normalize=False
):
    """
    Get 3D sinusoidal positional embedding (not learned).

    The input should be flattened into a N, 3 array.
    """
    # https://github.com/megvii-research/PETR/blob/main/projects/mmdet3d_plugin/models/dense_heads/petr_head.py#L29
    scale = 2 * math.pi
    pos = pos * scale

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None]
    pos_y = pos[..., 1, None]
    pos_z = pos[..., 2, None]
    if normalize:
        eps = 1e-6
        pos_x = (pos_x - 0.5) / (pos_x[-1] + eps) * scale
        pos_y = (pos_y - 0.5) / (pos_y[-1] + eps) * scale
        pos_z = (pos_z - 0.5) / (pos_z[-1] + eps) * scale

    pos_x = pos_x / dim_t
    pos_y = pos_y / dim_t
    pos_z = pos_z / dim_t

    pos_x = torch.stack(
        (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    pos_y = torch.stack(
        (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    pos_z = torch.stack(
        (pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1).permute(1, 0)
    return posemb


def get_3d_sinusoidal_pos_plus_vessel_emb(
    pos,
    vessel_dist,
    src_patch_size=(64, 64, 64),
    num_pos_feats=256,
    temperature=10000,
    normalize=False,
):
    # https://github.com/megvii-research/PETR/blob/main/projects/mmdet3d_plugin/models/dense_heads/petr_head.py#L29

    scale = 2 * math.pi

    pos_v = (
        vessel_dist[:, :, pos[:, 0].long(), pos[:, 1].long(), pos[:, 2].long()] * scale
    )
    pos = pos * scale

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None].unsqueeze(0).repeat(vessel_dist.shape[0], 1, 1)
    pos_y = pos[..., 1, None].unsqueeze(0).repeat(vessel_dist.shape[0], 1, 1)
    pos_z = pos[..., 2, None].unsqueeze(0).repeat(vessel_dist.shape[0], 1, 1)
    pos_v = pos_v.permute(0, 2, 1)

    if normalize:
        eps = 1e-6
        pos_x = (pos_x - 0.5) / (pos_x.max() + eps) * scale
        pos_y = (pos_y - 0.5) / (pos_y.max() + eps) * scale
        pos_z = (pos_z - 0.5) / (pos_z.max() + eps) * scale
        # vessel embedding is normalized w.r.t. patch size for consistency
        pos_v = (pos_v - 0.5) / (src_patch_size[0] + eps) * scale

    pos_x = pos_x / dim_t
    pos_y = pos_y / dim_t
    pos_z = pos_z / dim_t
    pos_v = pos_v / dim_t

    pos_x = torch.stack(
        (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    pos_y = torch.stack(
        (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    pos_z = torch.stack(
        (pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    pos_v = torch.stack(
        (pos_v[..., 0::2].sin(), pos_v[..., 1::2].cos()), dim=-1
    ).flatten(-2)

    posemb = torch.cat((pos_y, pos_x, pos_z, pos_v), dim=-1).permute(0, 2, 1)

    return posemb


def create_global_pos_volume(depth, height, width):
    """
    Given the size of the volume, create a 3D array where
    each element is the coordinate of the element in the volume.

    Parameters
    __________
    depth: int
        depth of the volume
    height: int
        height of the volume
    width: int
        width of the volume

    Returns
    _______
    volume_coords: torch.Tensor
        Tensor of shape (depth, height, width) where each element is the
        coordinate of the element in the volume.

    """
    indices = np.indices((depth, height, width))
    arr = np.transpose(indices, (1, 2, 3, 0)).reshape(-1, 3)
    volume_coords = torch.from_numpy(arr).long()
    return volume_coords
