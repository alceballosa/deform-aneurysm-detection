import torch
from torch import nn

from src.utils.general import get_activation_fn, get_clones, inverse_sigmoid


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_heads=8,
    ):
        super().__init__()

        # cross attention
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
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
        global_feats,
        global_pos_embed,
        key_padding_mask=None,
    ):
        # self attention
        q = k = self.with_pos_embed(ref, ref_pos_embed)
        ref2 = self.self_attn(
            q.transpose(0, 1), k.transpose(0, 1), ref.transpose(0, 1)
        )[0].transpose(0, 1)
        ref = ref + self.dropout2(ref2)
        ref = self.norm2(ref)

        # cross attention
        q = self.with_pos_embed(ref, ref_pos_embed)
        k = self.with_pos_embed(global_feats, global_pos_embed)

        ref2 = self.cross_attn(
            query=q.transpose(0, 1),
            key=k.transpose(0, 1),
            value=k.transpose(0, 1),
            key_padding_mask=key_padding_mask,
        )[0].transpose(0, 1)

        ref = ref + self.dropout1(ref2)
        ref = self.norm1(ref)

        # ffn
        ref = self.forward_ffn(ref)

        return ref


class TransformerDecoder(nn.Module):
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
        query,
        ref_loc,
        query_pos_embed,
        global_feats,
        global_pos_embed,
        key_padding_mask=None,
    ):
        output = query

        intermediate = []
        prev_ref_loc = ref_loc
        intermediate_ref_locs = []
        box_prediction_list = []
        viz_outputs_list = []
        for lid, layer in enumerate(self.layers):
            output = layer(
                output,
                query_pos_embed,
                global_feats,
                global_pos_embed,
                key_padding_mask,
            )

            viz_outputs_list.append({})

            if lid == self.num_layers - 1 or self.with_stepwise_loss:  # lastlayer
                prev_ref_loc = ref_loc
                ref_loc = self.refine_center(lid, ref_loc, output)
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

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_ref_locs.append(ref_loc)

        if self.return_intermediate:
            ref_loc = torch.stack(intermediate_ref_locs)

        return box_prediction_list, viz_outputs_list
