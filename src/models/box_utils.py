import math

import numpy as np
import torch


def nms_3D(dets, overlap=0.5, top_k=200):
    # det {prob, ctr_z, ctr_y, ctr_x, d, h, w}
    dd, hh, ww = dets[:, 4], dets[:, 5], dets[:, 6]
    z1 = dets[:, 1] - 0.5 * dd
    y1 = dets[:, 2] - 0.5 * hh
    x1 = dets[:, 3] - 0.5 * ww
    z2 = dets[:, 1] + 0.5 * dd
    y2 = dets[:, 2] + 0.5 * hh
    x2 = dets[:, 3] + 0.5 * ww
    scores = dets[:, 0]
    areas = dd * hh * ww
    _, idx = scores.sort(0, descending=True)
    keep = []
    while idx.size(0) > 0:
        i = idx[0]
        keep.append(int(i.cpu().numpy()))
        if idx.size(0) == 1 or len(keep) == top_k:
            break
        xx1 = torch.max(x1[idx[1:]], x1[i].expand(len(idx) - 1))
        yy1 = torch.max(y1[idx[1:]], y1[i].expand(len(idx) - 1))
        zz1 = torch.max(z1[idx[1:]], z1[i].expand(len(idx) - 1))

        xx2 = torch.min(x2[idx[1:]], x2[i].expand(len(idx) - 1))
        yy2 = torch.min(y2[idx[1:]], y2[i].expand(len(idx) - 1))
        zz2 = torch.min(z2[idx[1:]], z2[i].expand(len(idx) - 1))

        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)
        d = torch.clamp(zz2 - zz1, min=0.0)

        inter = w * h * d
        IoU = inter / (areas[i] + areas[idx[1:]] - inter)
        inds = IoU <= overlap
        idx = idx[1:][inds]
    return torch.from_numpy(np.array(keep))


def iou_3D(box1, box2):
    # need z_ctr, y_ctr, x_ctr, d
    z1 = np.maximum(box1[0] - 0.5 * box1[3], box2[0] - 0.5 * box2[3])
    y1 = np.maximum(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
    x1 = np.maximum(box1[2] - 0.5 * box1[3], box2[2] - 0.5 * box2[3])

    z2 = np.minimum(box1[0] + 0.5 * box1[3], box2[0] + 0.5 * box2[3])
    y2 = np.minimum(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3])
    x2 = np.minimum(box1[2] + 0.5 * box1[3], box2[2] + 0.5 * box2[3])

    w = np.maximum(x2 - x1, 0.0)
    h = np.maximum(y2 - y1, 0.0)
    d = np.maximum(z2 - z1, 0.0)

    inters = w * h * d
    uni = box1[3] * box1[3] * box1[3] + box2[3] * box2[3] * box2[3] - inters
    uni = np.maximum(uni, 1e-8)
    ious = inters / uni
    return ious


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
