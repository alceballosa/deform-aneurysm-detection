"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
This file is derived from [DETR](https://github.com/facebookresearch/detr/blob/main/models/transformer.py).
Modified for [PARQ] by Yiming Xie.

Original header:
Copyright 2020 - present, Facebook, Inc

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

DETR Transformer class.
Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
# from flash_attn.modules.mha import MHA
from src.models.parq.box_processor import BoxProcessor
from src.models.parq.deformable_attention import DeformableAttention3D
from src.models.parq.parq_utils import get_3d_sinusoidal_pos_emb, inverse_sigmoid
from torch import Tensor, nn
from torch.nn.functional import grid_sample


class Transformer_DA(nn.Module):
    def __init__(
        self,
        dec_dim=512,
        queries_dim=512,
        dec_heads=8,
        dec_layers=6,
        dec_ffn_dim=2048,
        dropout_rate=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=True,
        scale=None,
        share_weights=False,
        use_pe="use_pe",
    ):
        super().__init__()
        assert (
            queries_dim == dec_dim
        ), f"queries dim {queries_dim} needs to be equal to input enc_dim {dec_dim} for transformer encoder"
        self.d_model = dec_dim
        self.nhead = dec_heads

        decoder_layer = TransformerDecoderLayer(
            dec_dim,
            dec_heads,
            dec_ffn_dim,
            dropout_rate,
            activation,
            normalize_before,
            use_pe=use_pe,
        )
        decoder_norm = nn.LayerNorm(dec_dim)
        self.decoder = TransformerDecoder(
            decoder_layer,
            dec_layers,
            dec_dim,
            scale,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
            share_weights=share_weights,
            use_pe=use_pe,
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, global_tokens, query_ref_init, pre_tokenization_shape, mask=None):
        """
        Decoder - PARQ module, including transformer decoder and mlp heads
        input:
            global_tokens:       tensor, (B, N, C), the output of tokenization of (image features + ray positional encoding)
            query_tokens:       tensor, (N', C), reference points, N' is the number of reference points
        output:
            output_list (parameters of 3d boxes):
            [
                iteration 1 (dict):
                    pred_logits:          (B, N, num_semcls + 1), the classification logits (including background class)
                    center_unnormalized:  (B, N, 3), the predicted center of the box
                    size_unnormalized:    (B, N, 3), the predicted size of the box
                    sem_cls_prob:         (B, N, num_semcls + 1), the softmax of pred_logits
                    coord_pos:            (B, N, 3), the position of reference points, used for matcher
                iteration 2 (dict):
                    ...
                ...
            ]
        """
        B = global_tokens.shape[0]
        query_ref_init = query_ref_init.repeat(B, 1, 1)

        output_list, attn_list = self.decoder.forward(
            None,
            global_tokens,
            global_tokens_key_padding_mask=mask,
            query_pos_init=query_ref_init,
            pre_tokenization_shape=pre_tokenization_shape,
        )
        return output_list, attn_list


def retrieve_query_features(global_tokens_vol, query_pos, neighborhood_size=3):
    """
    Project reference points onto multi-view image to fetch appearence features
    Bilinear interpolation is used to fetch features.
    Average pooling is used to aggregate features from different views.
    """

    assert neighborhood_size % 2 == 1, "neighborhood size must be odd"
    # TODO: instead of casting to int, try to sample with img grid_sample
    # TODO: make this work with neighborhoods
    B, C, T, H, W = global_tokens_vol.shape
    B, N, _ = query_pos.shape
    query_tokens = []
    query_tokens = torch.empty(B, N, C).to(global_tokens_vol.device)
    global_tokens_vol = global_tokens_vol.permute(0, 2, 3, 4, 1)
    for b in range(B):
        for n in range(N):
            query_tokens[b, n, :] = global_tokens_vol[
                b, query_pos[b, n, 0], query_pos[b, n, 1], query_pos[b, n, 2], :
            ]
    return query_tokens


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers,
        dim_in,
        scale,
        norm=None,
        return_intermediate=False,
        share_weights=False,
        tokenization_scale_factor=(1, 1, 1),  # TODO: from yaml URGENT
        use_pe="use_pe",
    ):
        super().__init__()
        if not share_weights:
            self.layers = _get_clones(decoder_layer, num_layers)
        else:
            self.layers = _get_clones(decoder_layer, 1)
        self.num_layers = num_layers
        self.use_pe = use_pe

        self.share_weights = share_weights
        # self.norm = norm
        self.return_intermediate = return_intermediate
        self.position_ff_encoder = torch.nn.Sequential(
            torch.nn.Linear(128 * 3, dim_in),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_in, dim_in),
        )
        self.tokenization_scale_factor = tokenization_scale_factor
        self.scale = scale
        self.mlp_heads: nn.ModuleDict
        self.box_processor: BoxProcessor

    def normalize(self, points, rescaling_dims):
        """
        points: (B, N, 3)
        """
        points_1 = points[..., 0] / rescaling_dims[0]
        points_2 = points[..., 1] / rescaling_dims[1]
        points_3 = points[..., 2] / rescaling_dims[2]
        points = torch.stack([points_1, points_2, points_3], dim=-1)
        return points

    def denormalize(self, points, rescaling_dims):
        """
        points: (B, N, 3)
        """
        points_1 = points[..., 0] * rescaling_dims[0]
        points_2 = points[..., 1] * rescaling_dims[1]
        points_3 = points[..., 2] * rescaling_dims[2]
        points = torch.stack([points_1, points_2, points_3], dim=-1)

        return points

    def bbox3d_prediction(self, tokens, query_pos, dims, layer_num):
        """
        Predict the paramers of the boxes via multiple MLP heads
        input:
            tokens: tensor, (B, N, C), output of transformer decoder
            query_pos: tensor, (B, N, 3), coordinates of the reference points
            layer_num: int, the layer number of the transformer decoder
        output:
            out_dict: parameters of 3d boxes
        """
        if tokens.dim() == 4:
            tokens_list = torch.split(tokens, 1, dim=0)
        else:
            tokens_list = [tokens.unsqueeze(0)]

        share_mlp_heads = not isinstance(
            self.mlp_heads["sem_cls_head"], torch.nn.ModuleList
        )

        box_prediction_list = []
        for tokens in tokens_list:
            # TODO:w hy?
            tokens = tokens[0]
            # tokens are B x nqueries x noutput, change to B x noutput x bqueries
            tokens = tokens.permute(0, 2, 1).contiguous()
            if share_mlp_heads:
                cls_logits = self.mlp_heads["sem_cls_head"](tokens).transpose(1, 2)
                center_offset = self.mlp_heads["center_head"](tokens).transpose(1, 2)
            else:
                cls_logits = self.mlp_heads["sem_cls_head"][layer_num](
                    tokens
                ).transpose(1, 2)
                center_offset = self.mlp_heads["center_head"][layer_num](
                    tokens
                ).transpose(1, 2)
                # TODO: urgent, verify if this fucks up training

            query_pos_unnormalized = self.denormalize(query_pos, dims)
            query_pos_with_offset = center_offset + inverse_sigmoid(query_pos)
            center_normalized = query_pos_with_offset.sigmoid()
            center_unnormalized = self.denormalize(center_normalized, dims)

            if share_mlp_heads:
                size_normalized = (
                    self.mlp_heads["size_head"](tokens).transpose(1, 2).sigmoid()
                )
            else:
                size_normalized = (
                    self.mlp_heads["size_head"][layer_num](tokens)  # type: ignore
                    .transpose(1, 2)
                    .sigmoid()
                )

            # below are not used in computing loss (only for matching/mAP eval)
            # we compute them with no_grad() so that distributed training does not complain about unused variables
            with torch.no_grad():
                (
                    sem_cls_prob,
                    objectness_prob,  # TODO: check t his
                ) = self.box_processor.compute_objectness_and_cls_prob(cls_logits)

            # size_unnormalized = self.box_processor.compute_predicted_size(
            #    size_scale, semcls_prob
            # )
            size_unnormalized = self.denormalize(size_normalized, dims)
            box_prediction = {
                "pred_logits": cls_logits,
                "center_unnormalized": center_unnormalized,
                "size_unnormalized": size_unnormalized,
                "sem_cls_prob": sem_cls_prob,
                # used in matcher
                # TODO: redundant coord pos and pred_logits? ASK YIMING AND CHANGE ON MATCHER FILE
                "coord_pos": query_pos_unnormalized,  # use input reference point to match instead of output center
            }
            box_prediction_list.append(box_prediction)
        return box_prediction_list

    def create_global_pos_volume(self, depth, height, width):
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

    def forward(
        self,
        query_tokens,
        global_tokens,
        pre_tokenization_shape: tuple,
        query_tokens_mask: Optional[Tensor] = None,
        global_tokens_mask: Optional[Tensor] = None,
        query_tokens_key_padding_mask: Optional[Tensor] = None,
        global_tokens_key_padding_mask: Optional[Tensor] = None,
        global_pos: Optional[
            Tensor
        ] = None,  # TODO: move global_pos to trx layer, don't create the volume every single time
        query_pos_init: Optional[Tensor] = None,
        return_attn: bool = False,
    ):
        """
        tgt: the target
        """

        dt, dh, dw = self.tokenization_scale_factor
        bs, _, t, h, w = pre_tokenization_shape

        # create and apply position embedding
        if global_pos is None and self.use_pe in ["use_pe", "only_pe", "only_once"]:
            global_pos = self.create_global_pos_volume(t // dt, h // dh, w // dw).to(
                global_tokens.device
            )
            global_pos = global_pos.unsqueeze(0).repeat(bs, 1, 1)
            global_pos_emb = self.position_ff_encoder(
                get_3d_sinusoidal_pos_emb(global_pos)
            )
            global_tokens = global_tokens + global_pos_emb
        else:
            global_pos_emb = None
        global_tokens_volume = global_tokens.view(
            bs, t // dt, h // dh, w // dw, -1
        ).permute(0, 4, 1, 2, 3)

        iteration_outputs = []
        attn_outputs = []

        # get initial query_pos from trainable parameter query_pos_init
        query_pos = query_pos_init.sigmoid()
        rescaling_dims = t // dt, h // dh, w // dw

        for layer_num in range(self.num_layers):
            if self.share_weights:
                layer: TransformerDecoderLayer = self.layers[0]  # type: ignore
            else:
                layer: TransformerDecoderLayer = self.layers[layer_num]  # type: ignore

            query_pos_emb = self.position_ff_encoder(
                get_3d_sinusoidal_pos_emb(query_pos)
            )

            # TODO: instead of casting to int, try to sample with img grid_sample
            query_tokens = retrieve_query_features(
                global_tokens_volume,
                self.denormalize(query_pos, rescaling_dims).long(),
            )

            attended_query_tokens, attn = layer.forward(
                query_tokens,
                global_tokens_volume,
                query_mask=query_tokens_mask,
                global_tokens_mask=global_tokens_mask,
                tgt_key_padding_mask=query_tokens_key_padding_mask,
                memory_key_padding_mask=global_tokens_key_padding_mask,
                global_pos_emb=global_pos_emb,
                query_pos_emb=query_pos_emb,
            )
            output_dict = self.bbox3d_prediction(
                attended_query_tokens, query_pos, rescaling_dims, layer_num
            )

            query_pos = self.normalize(
                output_dict[0]["center_unnormalized"], rescaling_dims
            )

            query_pos = query_pos.detach()
            # output_dict = self.rescale_positions_and_sizes(output_dict, volume_to_token_size_ratio))

            if self.return_intermediate:
                iteration_outputs += output_dict

            attn_out = attn.detach().cpu().numpy() if return_attn else None
            attn_outputs.append(attn_out)
        return iteration_outputs, attn_outputs

    # def rescale_positions_and_sizes(self, output_dict, ratio):
    #     """
    #     Rescale the predicted center and size to the original scale
    #     input:
    #         output_dict: dict, the output of transformer decoder
    #         ratio: float, the ratio of original coordinate space
    #                 to feature coordinate space
    #     output:
    #         output_dict: dict, the output of transformer decoder
    #     """
    #     for i in range(len(output_dict)):
    #         output_dict[i]["center_unnormalized"] = (
    #             output_dict[i]["center_unnormalized"] * ratio
    #         )
    #         output_dict[i]["size_unnormalized"] = (
    #             output_dict[i]["size_unnormalized"] * ratio
    #         )
    #     return output_dict


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        use_pe="use_pe",
        use_flash_attn=False,
    ):
        super().__init__()
        self.use_flash_attn = use_flash_attn


        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.multihead_attn = DeformableAttention3D(
            dim=d_model,  # feature dimensions
            dim_head=d_model // nhead,  # dimension per head
            heads=nhead,  # attention heads
            dropout=dropout,  # dropout
            downsample_factor=(2, 8, 8),  # downsample factor (r in paper)
            offset_scale=(2, 8, 8),  # scale of offset, maximum offset
            offset_kernel_size=(4, 10, 10),  # offset kernel size
        )

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        # TODO: make into parameter from config
        self.use_pe = use_pe

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        if self.use_pe == "use_pe":
            return tensor if pos is None else tensor + pos
        elif self.use_pe in ["no_pe", "only_once"]:
            return tensor
        elif self.use_pe == "only_pe":
            return pos
        else:
            raise ValueError(f"invalid use_pe value {self.use_pe}")

    def forward_post(
        self,
        query_tokens,
        global_tokens,
        query_mask: Optional[Tensor] = None,
        global_tokens_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        global_tokens_key_padding_mask: Optional[Tensor] = None,
        global_pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(query_tokens, query_pos)
        # with torch.backends.cuda.sdp_kernel(
        #     enable_flash=False, enable_math=False, enable_mem_efficient=False
        # ):

        query_tokens_attended, attn = self.self_attn(
            q,
            k,
            value=query_tokens,
            attn_mask=query_mask,
            key_padding_mask=tgt_key_padding_mask,
        )

        query_tokens = query_tokens + self.dropout1(query_tokens_attended)
        query_tokens = self.norm1(query_tokens)

        query_tokens_attended, attn = self.multihead_attn(
            global_tokens,
            query_tokens,
        )
        query_tokens = query_tokens + self.dropout2(query_tokens_attended)
        query_tokens = self.norm2(query_tokens)
        query_tokens_attended = self.linear2(
            self.dropout(self.activation(self.linear1(query_tokens)))
        )
        query_tokens = query_tokens + self.dropout3(query_tokens_attended)
        query_tokens = self.norm3(query_tokens)
        return query_tokens, attn

    def forward_pre(
        self,
        query_tokens,
        global_tokens,
        query_mask: Optional[Tensor] = None,  # query
        global_tokens_mask: Optional[Tensor] = None,  # key
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        global_pos_emb: Optional[Tensor] = None,
        query_pos_emb: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(query_tokens)
        q = k = self.with_pos_embed(tgt2, query_pos_emb)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt2,
            attn_mask=query_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        query_tokens = query_tokens + self.dropout1(tgt2)
        tgt2 = self.norm2(query_tokens)
        tgt2, attn = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos_emb),
            key=self.with_pos_embed(global_tokens, global_pos_emb),
            value=global_tokens,
            attn_mask=global_tokens_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        query_tokens = query_tokens + self.dropout2(tgt2)
        tgt2 = self.norm3(query_tokens)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        query_tokens = query_tokens + self.dropout3(tgt2)
        return query_tokens, attn

    def forward(
        self,
        query_tokens,  # query
        global_tokens,  # key
        query_mask: Optional[Tensor] = None,
        global_tokens_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        global_pos_emb: Optional[Tensor] = None,
        query_pos_emb: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                query_tokens,
                global_tokens,
                query_mask,
                global_tokens_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                global_pos_emb,
                query_pos_emb,
            )
        else:
            return self.forward_post(
                query_tokens,
                global_tokens,
                query_mask,
                global_tokens_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                global_pos_emb,
                query_pos_emb,
            )


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


"""
    def forward_post_flash(
        self,
        query_tokens,
        global_tokens,
        query_mask: Optional[Tensor] = None,
        global_tokens_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        global_tokens_key_padding_mask: Optional[Tensor] = None,
        global_pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(query_tokens, query_pos)
        query_tokens_attended = self.self_attn(
            x=q,
            max_seqlen=q.shape[-1],
        )

        query_tokens = query_tokens + self.dropout1(query_tokens_attended)
        query_tokens = self.norm1(query_tokens)
        q = self.with_pos_embed(query_tokens, query_pos)
        k = self.with_pos_embed(global_tokens, global_pos)

        query_tokens_attended = self.multihead_attn(
            x=q,
            x_kv=k,
            max_seqlen=q.shape[-1],
        )
        query_tokens = query_tokens + self.dropout2(query_tokens_attended)
        query_tokens = self.norm2(query_tokens)
        query_tokens_attended = self.linear2(
            self.dropout(self.activation(self.linear1(query_tokens)))
        )
        query_tokens = query_tokens + self.dropout3(query_tokens_attended)
        query_tokens = self.norm3(query_tokens)
        return query_tokens
"""
