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

from functools import partial

import torch
from src.models.deformable.def_trx_decoder_rec import (
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
)
from src.models.deformable.def_trx_encoder import (
    DeformableTransformerEncoder,
    DeformableTransformerEncoderLayer,
)
from src.models.parq.generic_mlp import GenericMLP
from src.utils.general import get_clones
from torch import nn
from torch.nn.init import constant_, normal_, uniform_, xavier_uniform_


def build_deformable_transformer(cfg):
    center_head, size_head, class_head = build_heads(cfg)
    return Transformer(
        queries_dim=cfg.MODEL.D_MODEL,
        enc_dim=cfg.MODEL.D_MODEL,
        enc_heads=cfg.MODEL.DEFORMABLE.N_HEADS,
        n_enc_layers=cfg.MODEL.DEFORMABLE.N_ENC_LAYERS,
        enc_ffn_dim=cfg.MODEL.DEFORMABLE.FFN_DIM,
        n_enc_points=cfg.MODEL.DEFORMABLE.N_ENC_POINTS,
        dec_dim=cfg.MODEL.D_MODEL,
        dec_heads=cfg.MODEL.DEFORMABLE.N_HEADS,
        n_dec_layers=cfg.MODEL.DEFORMABLE.N_DEC_LAYERS,
        n_dec_points=cfg.MODEL.DEFORMABLE.N_DEC_POINTS,
        dec_ffn_dim=cfg.MODEL.DEFORMABLE.FFN_DIM,
        dropout_rate=cfg.MODEL.DEFORMABLE.DROPOUT,
        activation=cfg.MODEL.DEFORMABLE.ACTIVATION,
        use_global_pe=cfg.MODEL.DEFORMABLE.USE_GLOBAL_PE,
        n_levels=cfg.MODEL.DEFORMABLE.N_LEVELS,
        offset_init=cfg.MODEL.DEFORMABLE.OFFSET_INIT,
        use_fixed_attn=cfg.MODEL.DEFORMABLE.FIXED_ATTENTION,
        decoder_only=cfg.MODEL.DEFORMABLE.DECODER_ONLY,
        with_recurrence=cfg.MODEL.DEFORMABLE.WITH_RECURRENCE,
        with_stepwise_loss=cfg.MODEL.DEFORMABLE.WITH_STEPWISE_LOSS,
        shared_heads=cfg.MODEL.DEFORMABLE.SHARED_CENTER_HEAD,
        center_head=center_head,
        class_head=class_head,
        size_head=size_head,
    )


def build_heads(cfg, num_semcls=1):
    """
    Build mlp head to regress the loc/center of the queries/boxes
    """
    # TODO: check whether it is necessary to have the small mlp funcs.

    shared_heads = cfg.MODEL.DEFORMABLE.SHARED_CENTER_HEAD
    d_model = cfg.MODEL.D_MODEL
    n_layers = cfg.MODEL.DEFORMABLE.N_DEC_LAYERS
    mlp_dropout = cfg.MODEL.DEFORMABLE.HEAD_DROPOUT

    mlp_func = partial(
        GenericMLP,
        norm_fn_name="ln",
        activation="relu",
        use_conv=True,
        hidden_dims=[d_model, d_model],
        dropout=mlp_dropout,
        input_dim=d_model,
    )
    center_head = mlp_func(output_dim=3)
    size_head = mlp_func(output_dim=3)
    class_head = mlp_func(output_dim=num_semcls + 1)
    # TODO: apply init for heads, 239
    if shared_heads:
        return center_head, size_head, class_head
    else:
        return (
            get_clones(center_head, n_layers),
            get_clones(size_head, n_layers),
            get_clones(class_head, n_layers),
        )


""" from hieu
        # prior = 0.01
        # nn.init.constant_(self.head.cls_output.weight, 0)
        # nn.init.constant_(self.head.cls_output.bias, -math.log((1.0 - prior) / prior))

        # nn.init.constant_(self.head.shape_output.weight, 0)
        # nn.init.constant_(self.head.shape_output.bias, 0.5)

        # nn.init.constant_(self.head.offset_output.weight, 0)
        # nn.init.constant_(self.head.offset_output.bias, 0.05)
"""


class Transformer(nn.Module):
    def __init__(
        self,
        center_head,
        class_head,
        size_head,
        queries_dim=512,
        enc_dim=512,
        enc_heads=8,
        n_enc_layers=6,
        enc_ffn_dim=2048,
        n_enc_points=4,
        dec_dim=512,
        dec_heads=8,
        n_dec_layers=6,
        dec_ffn_dim=2048,
        n_dec_points=4,
        dropout_rate=0.1,
        activation="relu",
        n_levels=4,
        use_global_pe=False,
        offset_init="strict",
        decoder_only=False,
        with_recurrence=False,
        with_stepwise_loss=False,
        return_intermediate_dec=False,
        shared_heads=True,
        use_fixed_attn=False,
        # TODO
        # TODO: make this work, URGENT
    ):
        super().__init__()
        assert (
            queries_dim == dec_dim
        ), f"queries dim {queries_dim} needs to be equal to input enc dim {dec_dim}"

        # TODO assert that with_recurrence and shared_head cannot be false at the same time
        assert (
            with_recurrence or shared_heads
        ), "with_recurrence and shared_head cannot be false at the same time"

        self.n_levels = n_levels
        self.decoder_only = decoder_only
        self.level_embed = nn.Parameter(torch.Tensor(n_levels, dec_dim))
        self.use_global_pe = use_global_pe
        self.reference_points = nn.Linear(dec_dim, 3)

        if not decoder_only:
            encoder_layer = DeformableTransformerEncoderLayer(
                enc_dim,
                enc_ffn_dim,
                dropout_rate,
                activation,
                n_levels,
                enc_heads,
                n_enc_points,
            )
            self.encoder = DeformableTransformerEncoder(encoder_layer, n_enc_layers)

        decoder_layer = DeformableTransformerDecoderLayer(
            dec_dim,
            dec_ffn_dim,
            dropout_rate,
            activation,
            n_levels,
            dec_heads,
            n_dec_points,
            offset_init,
            use_fixed_attn
        )
        self.decoder = DeformableTransformerDecoder(
            decoder_layer,
            n_dec_layers,
            with_recurrence,
            with_stepwise_loss,
            return_intermediate=return_intermediate_dec,
            shared_heads=shared_heads,
        )

        self.decoder.center_head = center_head
        self.decoder.class_head = class_head
        self.decoder.size_head = size_head

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        # NOTE: additional init 239
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.0)
        normal_(self.level_embed)

    def forward(
        self,
        multiscale_feats,
        multiscale_pos_embs,
        ref_pos_embed_plus_feat,
    ):
        """
        Transformer module for multiscale deformable attention in PARQ.

        Parameters
        __________

        multiscale_feats: List[torch.Tensor]
            List of features from different levels of the UNet backbone.
            Each feature map has shape [B, C, D, H, W], where C is expected to be
            the same for all levels.
        multiscale_pos_embs: List[torch.Tensor]
            List of position embeddings for different levels of the backbone, flattened.
            The shape is expected to be [N, C], where N = D*H*W for the particular level.

        """
        # B = global_tokens.shape[0]
        # query_ref_init = query_ref_init.repeat(B, 1, 1)

        # output_list, attn_list = self.decoder.forward(
        #     None,
        #     global_tokens,
        #     global_tokens_key_padding_mask=mask,
        #     query_pos_init=query_ref_init,
        #     pre_tokenization_shape=pre_tokenization_shape,
        # )

        bs = multiscale_feats[0].shape[0]

        extracted_feats_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, pos_embed) in enumerate(
            zip(multiscale_feats, multiscale_pos_embs)
        ):
            _, _, d, h, w = feat.shape
            spatial_shape = d, h, w
            spatial_shapes.append(spatial_shape)
            # flatten feat from B C D H W into B DHW C
            extracted_feats_flatten.append(feat.flatten(2).permute(0, 2, 1))
            if len(pos_embed.shape) == 3:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].unsqueeze(
                    1
                ).unsqueeze(0)
            else:  # when pos embed shape is len 2
                lvl_pos_embed = (
                    pos_embed + self.level_embed[lvl].unsqueeze(1)
                ).unsqueeze(0)
                
            lvl_pos_embed_flatten.append(lvl_pos_embed.permute(0, 2, 1))

        extracted_feats_flatten = torch.cat(extracted_feats_flatten, dim=1)

        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, dim=1)

        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=extracted_feats_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )

        if self.decoder_only:
            global_feats = extracted_feats_flatten
        else:
            global_feats = self.encoder(
                extracted_feats_flatten,
                spatial_shapes,
                level_start_index,
                lvl_pos_embed_flatten,
            )
        #
        bs, _, c = global_feats.shape
        ref_pos_embed, ref = torch.split(ref_pos_embed_plus_feat, c, dim=1)
        ref_pos_embed = ref_pos_embed.unsqueeze(0).expand(bs, -1, -1)
        ref = ref.unsqueeze(0).expand(bs, -1, -1)
        ref_loc = self.reference_points(ref_pos_embed).sigmoid()
        init_reference_out = ref_loc
        # decoder
        box_predictions, viz_outputs = self.decoder(
            ref,
            ref_loc,
            ref_pos_embed,
            global_feats,
            lvl_pos_embed_flatten if self.use_global_pe else None,
            spatial_shapes,
            level_start_index,
        )
        return box_predictions, init_reference_out, viz_outputs, None
