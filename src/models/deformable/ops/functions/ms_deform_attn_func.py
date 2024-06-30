# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import, division, print_function

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

# import MultiScaleDeformableAttention as MSDA


class MSDeformAttnFunction(Function):
    @staticmethod
    def forward(
        ctx,
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            ctx.im2col_step,
        )
        ctx.save_for_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        ) = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = MSDA.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output,
            ctx.im2col_step,
        )

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def ms_deform_3d_attn_core_pytorch(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    # for debug and test only,
    # need to use cuda version instead

    bs, value_len, n_heads, c_per_head = value.shape
    _, query_len, n_heads, n_levels, n_points, _ = sampling_locations.shape
    value_list = value.split(
        [depth * height * width for depth, height, width in value_spatial_shapes], dim=1
    )
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (depth, height, width) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        # bs, depth*height*width, n_heads, c_per_head
        # bs, depth*height*width, c
        # bs, c, depth*height*width
        # bs*n_heads, c_per_head, depth, height, width
        value_l_ = (
            value_list[lid_]
            .flatten(2)
            .transpose(1, 2)
            .reshape(bs * n_heads, c_per_head, depth, height, width)
        )
        # print("Value at level 0 re shaped", value_l_.shape)
        # bs, query_len, n_heads, n_points, 3
        # bs, n_heads, query_len, n_points, 3
        # bs*n_heads, query_len, n_points, 3
        # bs*n_heads, query_len//2, 2, n_points, 3
        sampling_grid_l_ = (
            sampling_grids[:, :, :, lid_]
            .transpose(1, 2)
            .flatten(0, 1)
            .view(bs * n_heads, query_len // 2, 2, n_points, 3)
        )

        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).view(bs * n_heads, c_per_head, query_len, n_points)

        sampling_value_list.append(sampling_value_l_)

    stacked_sampled_values = torch.stack(sampling_value_list, dim=-2).flatten(-2)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * n_heads, 1, query_len, n_levels * n_points
    )
    output = (
        (stacked_sampled_values * attention_weights)
        .sum(-1)
        .view(bs, n_heads * c_per_head, query_len)
    )
    return output.transpose(1, 2).contiguous()
