# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import, division, print_function

import math
import warnings

import torch
import torch.nn.functional as F
from src.models.deformable.ops.functions import (
    MSDeformAttnFunction,
    ms_deform_3d_attn_core_pytorch,
)
from torch import nn
from torch.nn.init import constant_, xavier_uniform_


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(
            "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n))
        )
    return (n & (n - 1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(
        self, d_model=256, n_levels=4, n_heads=8, n_points=4, offset_init="strict"
    ):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads, but got {} and {}".format(
                    d_model, n_heads
                )
            )
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation."
            )

        self.im2col_step = 64
        self.offset_init = offset_init
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 3)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)

        if self.n_heads != 16:
            # shameless hack
            thetas = torch.arange(self.n_heads, dtype=torch.float32) * (
                2.0 * math.pi / self.n_heads
            )
            grid_init = torch.stack([thetas.cos(), thetas.sin(), thetas.cos()], -1)
            grid_init = (
                (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
                .view(self.n_heads, 1, 1, 3)
                .repeat(1, self.n_levels, self.n_points, 1)
            )

            for i in range(self.n_points):
                grid_init[:, :, i, :] *= i + 1
        elif self.offset_init in ["strict", "relaxed"]:
            print(f"Using {self.offset_init} init in 3D")
            # sample uniformly from unit sphere 
            # https://math.stackexchange.com/questions/3184449/is-there-a-way-to-generate-individual-uniformly-distributed-points-on-a-sphere-f
            grid = torch.zeros((4, 4, 2))
            for i in range(4):
                for j in range(4):
                    grid[i, j, 0] = i
                    grid[i, j, 1] = j
            grid = grid / 4
            u_1 = grid[:, :, 0].flatten()
            u_2 = grid[:, :, 1].flatten()
            z = 2 * u_1 - 1
            x = torch.sqrt(1 - z**2) * torch.cos(2 * math.pi * u_2)
            y = torch.sqrt(1 - z**2) * torch.sin(2 * math.pi * u_2)
            grid_init = torch.stack(
                [z, x, y],
                -1,
            )
            grid_init = (
                (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
                .view(self.n_heads, 1, 1, 3)
                .repeat(1, self.n_levels, self.n_points, 1)
            )

            for i in range(self.n_points):
                grid_init[:, :, i, :] *= i + 1
        elif self.offset_init == "random":
            grid_init = torch.empty(16, 4, 32, 3)
            xavier_uniform_(grid_init) 

        if self.offset_init == "strict":
            with torch.no_grad():
                self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        else:
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
    ):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        bs, Len_q, _ = query.shape
        bs, Len_in, _ = input_flatten.shape
        assert (
            input_spatial_shapes[:, 0]
            * input_spatial_shapes[:, 1]
            * input_spatial_shapes[:, 2]
        ).sum() == Len_in
        value = self.value_proj(input_flatten)
        value = value.view(bs, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            bs, Len_q, self.n_heads, self.n_levels, self.n_points, 3
        )
        attention_weights = self.attention_weights(query).view(
            bs, Len_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            bs, Len_q, self.n_heads, self.n_levels, self.n_points
        )
        # N, Len_q, n_heads, n_levels, n_points, 2
        # TODO: continue from here
        # NOTE: yeah
        if reference_points.shape[-1] == 3:
            offset_normalizer = torch.stack(
                [
                    input_spatial_shapes[..., 0],
                    input_spatial_shapes[..., 1],
                    input_spatial_shapes[..., 2],
                ],
                -1,
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 3, but get {} instead.".format(
                    reference_points.shape[-1]
                )
            )
        # output = MSDeformAttnFunction.apply(
        #    value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = ms_deform_3d_attn_core_pytorch(
            value, input_spatial_shapes, sampling_locations, attention_weights
        )

        output = self.output_proj(output)
        return output, (sampling_locations,sampling_offsets,offset_normalizer,reference_points,input_spatial_shapes), attention_weights
