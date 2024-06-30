from typing import Union

import torch
from src.models.deformable.ops.modules import MSDeformAttn, MSDeformAttnFix
from src.utils.general import get_activation_fn, get_clones, inverse_sigmoid
from torch import nn


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
        offset_init="strict",
        use_fixed_attn=False,
    ):
        super().__init__()

        # cross attention
        deform_attn_cls = MSDeformAttnFix if use_fixed_attn else MSDeformAttn
        self.cross_attn = deform_attn_cls(
            d_model, n_levels, n_heads, n_points, offset_init
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(x, pos):

        return x if pos is None else x + pos

    def forward_ffn(self, x):
        x2 = self.linear2(self.dropout3(self.activation(self.linear1(x))))
        x = x + self.dropout4(x2)
        x = self.norm3(x)
        return x

    def forward(
        self,
        ref,
        ref_pos_embed,
        ref_loc,
        global_feats,
        global_pos_embed,
        global_feats_spatial_shapes,
        level_start_index,
    ):
        # self attention
        q = k = self.with_pos_embed(ref, ref_pos_embed)
        ref2 = self.self_attn(
            q.transpose(0, 1), k.transpose(0, 1), ref.transpose(0, 1)
        )[0].transpose(0, 1)
        ref = ref + self.dropout2(ref2)
        ref = self.norm2(ref)
        
        # cross attention
        ref2, sampling_locations, attn_weights = self.cross_attn(
            self.with_pos_embed(ref, ref_pos_embed),
            ref_loc,
            self.with_pos_embed(global_feats, global_pos_embed),
            global_feats_spatial_shapes,
            level_start_index,
        )
        ref = ref + self.dropout1(ref2)
        ref = self.norm1(ref)

        # ffn
        ref = self.forward_ffn(ref)

        return ref, sampling_locations, attn_weights


class DeformableTransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers,
        with_recurrence=False,
        with_stepwise_loss=False,
        shared_heads=False,
        return_intermediate=False,
    ):
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.with_recurrence = with_recurrence
        self.with_stepwise_loss = with_stepwise_loss
        self.bbox_embed = None
        self.class_embed = None
        self.shared_heads = shared_heads
        self.center_head: torch.nn.Module
        self.class_head: torch.nn.Module
        self.size_head: torch.nn.Module

    def select_center_head(self, layer):
        if self.shared_heads:
            return self.center_head
        else:
            return self.center_head[layer]  # type: ignore

    def select_class_head(self, layer):
        if self.shared_heads:
            return self.class_head
        else:
            return self.class_head[layer]  # type: ignore

    def select_size_head(self, layer):
        if self.shared_heads:
            return self.size_head
        else:
            return self.size_head[layer]  # type: ignore

    def refine_center(self, lid, reference_points, output):
        tmp = self.select_center_head(lid)(output.permute(0, 2, 1)).permute(0, 2, 1)
        new_reference_points = tmp + inverse_sigmoid(reference_points)
        new_reference_points = new_reference_points.sigmoid()
        return new_reference_points

    def forward(
        self,
        ref,
        ref_loc,
        ref_pos_embed,
        global_feats,
        global_pos_embed,
        src_spatial_shapes,
        src_level_start_index,
    ):
        output = ref

        intermediate = []
        prev_ref_loc = ref_loc
        intermediate_ref_locs = []
        box_prediction_list = []
        viz_outputs_list = []
        attn_weight_list = []
        for lid, layer in enumerate(self.layers):
            # if ref_loc.shape[-1] == 3:
            #     ref_loc_input = ref_loc[:, :, None]
            # else:
            #     raise ValueError(
            #         "Last dim of reference_points must be 3,  got {} instead.".format(
            #             ref_loc.shape[-1]
            #         )
            #     )
            output, sampling_locations, attn_weights = layer(
                output,
                ref_pos_embed,
                ref_loc[:, :, None],
                global_feats,
                global_pos_embed,
                src_spatial_shapes,
                src_level_start_index,
            )

            viz_outputs_list.append(
                {
                    "spatial_shapes": sampling_locations[4].to("cpu").detach().numpy(),
                    "offset_normalizer": sampling_locations[2].to("cpu").detach().numpy(),
                    "reference_points": sampling_locations[3].to("cpu").detach().numpy(),
                    "sampling_offsets": sampling_locations[1].to("cpu").detach().numpy(),
                    "sampling_locations": sampling_locations[0].to("cpu").detach().numpy(),
                    "pre_refinement_center": ref_loc.to("cpu")
                    .detach()
                    .numpy(),
                    "attn_weights": attn_weights.to("cpu").detach().numpy(),
                }
            )

            if self.with_recurrence or lid == self.num_layers - 1:
                prev_ref_loc = ref_loc
                ref_loc = self.refine_center(lid, ref_loc, output)

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_ref_locs.append(ref_loc)

            if lid == self.num_layers - 1 or self.with_stepwise_loss:  # lastlayer
                class_logits = self.select_class_head(lid)(
                    output.permute(0, 2, 1)
                ).permute(0, 2, 1)
                size = (
                    self.select_size_head(lid)(output.permute(0, 2, 1))
                    .permute(0, 2, 1)
                    .sigmoid()
                )
                class_probs = torch.nn.functional.softmax(class_logits, dim=-1)

                box_dict = {
                    "class_logits": class_logits,
                    "size": size,
                    "center": ref_loc,
                    "pre_refinement_center": prev_ref_loc,
                    "class_probs": class_probs,
                }
                box_prediction_list.append(box_dict)

        # if self.return_intermediate:
        #     return torch.stack(intermediate), torch.stack(intermediate_reference_points)
        if self.return_intermediate:
            ref_loc = torch.stack(intermediate_ref_locs)
        return box_prediction_list, viz_outputs_list
