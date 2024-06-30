import torch
from src.models.deformable.ops.modules import MSDeformAttn
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
    ):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
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
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        ref,
        ref_pos_embed,
        ref_loc,
        global_feats,
        src_spatial_shapes,
        level_start_index,
    ):
        # self attention
        q = k = self.with_pos_embed(ref, ref_pos_embed)
        tgt2 = self.self_attn(
            q.transpose(0, 1), k.transpose(0, 1), ref.transpose(0, 1)
        )[0].transpose(0, 1)
        ref = ref + self.dropout2(tgt2)
        ref = self.norm2(ref)

        # cross attention
        tgt2, offsets, attn_weights = self.cross_attn(
            self.with_pos_embed(ref, ref_pos_embed),
            ref_loc,
            global_feats,
            src_spatial_shapes,
            level_start_index,
        )
        ref = ref + self.dropout1(tgt2)
        ref = self.norm1(ref)

        # ffn
        ref = self.forward_ffn(ref)

        return ref


class DeformableTransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers,
        with_recurrence=False,
        shared_center_head=False,
        shared_decoder_layer_weights=False,
        return_intermediate=False,
    ):
        super().__init__()
        if shared_decoder_layer_weights:
            self.layers = nn.ModuleList([decoder_layer] * num_layers)
        else:
            self.layers = get_clones(decoder_layer, num_layers)

        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.with_recurrence = with_recurrence
        self.bbox_embed = None
        self.class_embed = None
        self.shared_center_head = shared_center_head

    def forward(
        self,
        tgt,
        reference_points,
        src,
        src_spatial_shapes,
        src_level_start_index,
        query_pos=None,
    ):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 3:
                reference_points_input = reference_points[:, :, None]
            else:
                raise ValueError(
                    "Last dim of reference_points must be 3, but get {} instead.".format(
                        reference_points.shape[-1]
                    )
                )
            output = layer(
                output,
                query_pos,
                reference_points_input,
                src,
                src_spatial_shapes,
                src_level_start_index,
            )

            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(
                        reference_points
                    )
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points
