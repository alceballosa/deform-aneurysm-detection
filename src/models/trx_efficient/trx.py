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

Efficient Transformer class for vessel-masked architecture.
Uses full self-attention encoder and standard cross-attention decoder
instead of deformable attention.
"""

from functools import partial

import torch
from src.models.trx_efficient.trx_decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
)
from src.models.trx_efficient.cond_trx_decoder import (
    ConditionalTransformerDecoder,
    ConditionalTransformerDecoderLayer,
)
from src.models.trx_efficient.trx_encoder import (
    TransformerEncoderLayer,
    DeformableTransformerEncoder,
)
from src.models.layers.generic_mlp import GenericMLP
from src.utils.general import get_clones
from torch import nn
from torch.nn.init import constant_, normal_, xavier_uniform_


def build_efficient_transformer(cfg):
    center_head, size_head, class_head = build_heads(cfg)
    return EfficientTransformer(
        queries_dim=cfg.MODEL.D_MODEL,
        enc_dim=cfg.MODEL.D_MODEL,
        enc_heads=cfg.MODEL.DEFORMABLE.N_HEADS,
        n_enc_layers=cfg.MODEL.DEFORMABLE.N_ENC_LAYERS,
        enc_ffn_dim=cfg.MODEL.DEFORMABLE.FFN_DIM,
        dec_dim=cfg.MODEL.D_MODEL,
        dec_heads=cfg.MODEL.DEFORMABLE.N_HEADS,
        n_dec_layers=cfg.MODEL.DEFORMABLE.N_DEC_LAYERS,
        dec_ffn_dim=cfg.MODEL.DEFORMABLE.FFN_DIM,
        dropout_rate=cfg.MODEL.DEFORMABLE.DROPOUT,
        activation=cfg.MODEL.DEFORMABLE.ACTIVATION,
        use_global_pe=cfg.MODEL.DEFORMABLE.USE_GLOBAL_PE,
        n_levels=cfg.MODEL.DEFORMABLE.N_LEVELS,
        decoder_only=cfg.MODEL.DEFORMABLE.DECODER_ONLY,
        with_recurrence=cfg.MODEL.DEFORMABLE.WITH_RECURRENCE,
        with_stepwise_loss=cfg.MODEL.DEFORMABLE.WITH_STEPWISE_LOSS,
        shared_heads=cfg.MODEL.DEFORMABLE.SHARED_CENTER_HEAD,
        center_head=center_head,
        class_head=class_head,
        size_head=size_head,
        use_checkpoint=cfg.MODEL.DEFORMABLE.USE_CHECKPOINT,
        use_conditional_decoder=cfg.MODEL.DEFORMABLE.USE_CONDITIONAL_DECODER,
    )


def build_heads(cfg, num_semcls=1):
    """
    Build mlp head to regress the loc/center of the queries/boxes
    """
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
    if shared_heads:
        return center_head, size_head, class_head
    else:
        return (
            get_clones(center_head, n_layers),
            get_clones(size_head, n_layers),
            get_clones(class_head, n_layers),
        )


class EfficientTransformer(nn.Module):
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
        dec_dim=512,
        dec_heads=8,
        n_dec_layers=6,
        dec_ffn_dim=2048,
        dropout_rate=0.1,
        activation="relu",
        n_levels=4,
        use_global_pe=False,
        decoder_only=False,
        with_recurrence=False,
        with_stepwise_loss=False,
        return_intermediate_dec=False,
        shared_heads=True,
        use_checkpoint=False,
        use_conditional_decoder=False,
    ):
        super().__init__()
        assert (
            queries_dim == dec_dim
        ), f"queries dim {queries_dim} needs to be equal to input enc dim {dec_dim}"

        assert (
            with_recurrence or shared_heads
        ), "with_recurrence and shared_head cannot be false at the same time"

        self.n_levels = n_levels
        self.decoder_only = decoder_only
        self.use_conditional_decoder = use_conditional_decoder
        self.level_embed = nn.Parameter(torch.Tensor(n_levels, dec_dim))
        self.use_global_pe = use_global_pe
        self.reference_points = nn.Linear(dec_dim, 3)

        if not decoder_only:
            encoder_layer = TransformerEncoderLayer(
                enc_dim,
                enc_ffn_dim,
                dropout_rate,
                activation,
                enc_heads,
            )
            self.encoder = DeformableTransformerEncoder(
                encoder_layer, n_enc_layers, use_checkpoint=use_checkpoint
            )

        if use_conditional_decoder:
            decoder_layer = ConditionalTransformerDecoderLayer(
                dec_dim,
                dec_ffn_dim,
                dropout_rate,
                activation,
                dec_heads,
            )
            self.decoder = ConditionalTransformerDecoder(
                decoder_layer,
                n_dec_layers,
                with_recurrence,
                with_stepwise_loss,
                return_intermediate=return_intermediate_dec,
                shared_heads=shared_heads,
            )
        else:
            decoder_layer = TransformerDecoderLayer(
                dec_dim,
                dec_ffn_dim,
                dropout_rate,
                activation,
                dec_heads,
            )
            self.decoder = TransformerDecoder(
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
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.0)
        normal_(self.level_embed)

    def forward(
        self,
        multiscale_feats,
        multiscale_pos_embs,
        ref_pos_embed_plus_feat,
        key_padding_mask=None,
    ):
        """
        Efficient transformer with vessel-masked full self-attention encoder.

        Parameters
        ----------
        multiscale_feats: List[torch.Tensor]
            Pre-tokenized features (already flattened/processed by backbone).
        multiscale_pos_embs: List[torch.Tensor]
            Position embeddings for the features.
        ref_pos_embed_plus_feat: torch.Tensor
            Concatenated reference position embeddings and features.
        key_padding_mask: optional
            Padding mask for keys.
        """

        if self.decoder_only:
            global_feats = multiscale_feats
        else:
            global_feats = self.encoder(
                multiscale_feats,
                multiscale_pos_embs,
                key_padding_mask,
            )

        bs, _, c = global_feats.shape
        ref_pos_embed, ref = torch.split(ref_pos_embed_plus_feat, c, dim=1)
        ref_pos_embed = ref_pos_embed.unsqueeze(0).expand(bs, -1, -1)
        ref = ref.unsqueeze(0).expand(bs, -1, -1)
        ref_loc = self.reference_points(ref_pos_embed).sigmoid()
        init_reference_out = ref_loc

        box_predictions, viz_outputs = self.decoder(
            ref,
            ref_loc,
            ref_pos_embed,
            global_feats,
            multiscale_pos_embs if self.use_global_pe else None,
            key_padding_mask,
        )
        return box_predictions, init_reference_out, viz_outputs, None
