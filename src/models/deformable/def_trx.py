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


import torch
from src.models.deformable.def_trx_decoder import (
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
)
from src.models.deformable.def_trx_encoder import (
    DeformableTransformerEncoder,
    DeformableTransformerEncoderLayer,
)
from torch import nn


def build_deformable_transformer(cfg):
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
        n_levels=cfg.MODEL.DEFORMABLE.N_LEVELS,
        decoder_only=cfg.MODEL.DEFORMABLE.DECODER_ONLY,
        with_recurrence=cfg.MODEL.DEFORMABLE.WITH_RECURRENCE,
        shared_decoder_layer_weights=cfg.MODEL.DEFORMABLE.SHARED_DECODER_LAYER_WEIGHTS,
    )


class Transformer(nn.Module):
    def __init__(
        self,
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
        decoder_only=False,
        with_recurrence=False,
        shared_decoder_layer_weights=False,
        return_intermediate_dec=False, # TODO
        # TODO: make this work, URGENT
    ):
        super().__init__()
        assert (
            queries_dim == dec_dim
        ), f"queries dim {queries_dim} needs to be equal to input enc dim {dec_dim}"

        self.n_levels = n_levels
        self.decoder_only = decoder_only
        self.level_embed = nn.Parameter(torch.Tensor(n_levels, dec_dim))

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
        )
        self.decoder = DeformableTransformerDecoder(
            decoder_layer, n_dec_layers, with_recurrence, return_intermediate=return_intermediate_dec
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, multiscale_feats, multiscale_pos_embs, query_embed):
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

        feats_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, pos_embed) in enumerate(
            zip(multiscale_feats, multiscale_pos_embs)
        ):
            _, _, d, h, w = feat.shape
            spatial_shape = d, h, w
            spatial_shapes.append(spatial_shape)
            # flatten feat from B C D H W into B DHW C
            feats_flatten.append(feat.flatten(2).permute(0, 2, 1))
            lvl_pos_embed = (pos_embed + self.level_embed[lvl].unsqueeze(1)).unsqueeze(
                0
            )
            lvl_pos_embed_flatten.append(lvl_pos_embed.permute(0, 2, 1))

        feats_flatten = torch.cat(feats_flatten, dim=1)

        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, dim=1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feats_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )

        if self.decoder_only:
            memory = feats_flatten
        else:
            memory = self.encoder(
                feats_flatten, spatial_shapes, level_start_index, lvl_pos_embed_flatten
            )
        #
        bs, _, c = memory.shape
        query_embed, tgt = torch.split(query_embed, c, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_embed).sigmoid()
        init_reference_out = reference_points
        # decoder
        hs, inter_references = self.decoder(
            tgt,
            reference_points,
            memory,
            spatial_shapes,
            level_start_index,
            query_embed,
        )
        attn = None
        return hs, init_reference_out, inter_references, attn
