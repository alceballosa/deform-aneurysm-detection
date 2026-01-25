"""
CNN Backbone class that implements a 3D CNN Encoder. This encoder only implements
the __init__ method defining the architecture and the encode_multiscale_feats method
that derives multiscale features from the backbone. The position embedding and forward
methods are implemented in the Base_Backbone class.
"""

import pdb
from typing import List

import torch
from torch import nn

from src.models.deformable.base_backbone import Base_Backbone
from src.models.layers.conv_layers import ConvBlock, DownsamplingConvBlock, LayerBasic


def build_backbone(cfg):
    """
    Builds the UNET Encoder for the PARQ model.
    """

    return CNN_Backbone(
        cfg,
        n_channels=cfg.DATA.N_CHANNELS,
        n_blocks=cfg.MODEL.CONV_MODEL.N_BLOCKS,
        n_filters=cfg.MODEL.CONV_MODEL.N_FILTERS,
        stem_filters=cfg.MODEL.CONV_MODEL.STEM_FILTERS,
        norm_type=cfg.MODEL.CONV_MODEL.NORM,
        act_type=cfg.MODEL.CONV_MODEL.ACT,
        se=cfg.MODEL.CONV_MODEL.SE,
        first_stride=cfg.MODEL.CONV_MODEL.FIRST_STRIDE,
        output_hidden_dim=cfg.MODEL.D_MODEL,
        n_levels=cfg.MODEL.DEFORMABLE.N_LEVELS,
    )


class CNN_Backbone(Base_Backbone):
    """
    UNET-based encoder for the PARQ model.
    """

    def __init__(
        self,
        cfg,
        n_channels=1,
        n_blocks=[2, 3, 3, 3],
        n_filters=[64, 96, 128, 160],
        stem_filters=32,
        norm_type="BN",
        act_type="ReLU",
        se=False,
        first_stride=(2, 2, 2),
        output_hidden_dim=768,
        n_levels=4,
    ):
        super(CNN_Backbone, self).__init__(cfg)

        assert n_levels >= 1 and n_levels <= 4
        # TODO: make levels matter

        self.cfg = cfg
        self.output_hidden_dim = output_hidden_dim
        self.n_levels = n_levels

        # # pretrained UNET
        # self.frozen_parameters_list = []

        self.in_conv = ConvBlock(
            n_channels, stem_filters, stride=1, norm_type=norm_type, act_type=act_type
        )
        self.in_dw = ConvBlock(
            stem_filters,
            n_filters[0],
            stride=first_stride,
            norm_type=norm_type,
            act_type=act_type,
        )

        self.block1 = LayerBasic(
            n_blocks[0],
            n_filters[0],
            n_filters[0],
            norm_type=norm_type,
            act_type=act_type,
            se=se,
        )
        self.block1_dw = DownsamplingConvBlock(
            n_filters[0], n_filters[1], norm_type=norm_type, act_type=act_type
        )

        self.block2 = LayerBasic(
            n_blocks[1],
            n_filters[1],
            n_filters[1],
            norm_type=norm_type,
            act_type=act_type,
            se=se,
        )
        self.block2_dw = DownsamplingConvBlock(
            n_filters[1], n_filters[2], norm_type=norm_type, act_type=act_type
        )

        self.block3 = LayerBasic(
            n_blocks[2],
            n_filters[2],
            n_filters[2],
            norm_type=norm_type,
            act_type=act_type,
            se=se,
        )
        self.block3_dw = DownsamplingConvBlock(
            n_filters[2], n_filters[3], norm_type=norm_type, act_type=act_type
        )

        self.block4 = LayerBasic(
            n_blocks[3],
            n_filters[3],
            n_filters[3],
            norm_type=norm_type,
            act_type=act_type,
            se=se,
        )

        self.layer_hidden_dims = [
            n_filters[0],
            n_filters[1],
            n_filters[2],
            n_filters[3],
        ]
        input_proj_list = []
        for i, _ in enumerate(self.layer_hidden_dims):
            in_channels = self.layer_hidden_dims[i]
            input_proj_list.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, self.output_hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.output_hidden_dim),
                )
            )
        self.input_proj_list = nn.ModuleList(input_proj_list)
        # NOTE: additional init 239
        for proj in self.input_proj_list:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def get_multiscale_mask(self, segs):
        """
        Generates a multiscale mask to only process regions with vessels.
        To ensure we don't miss areas where vessels are not properly  segmented,
        we do not mask the lowest resolution feature map (optional)
        """

        masks = []
        for i, seg in enumerate(segs):
            mask = seg.clone()
            if i == len(segs) - 1 and self.cfg.MODEL.DEFORMABLE.STRICT_MASKING is False:
                # no masking at the lowest resolution
                # if strict masking is False
                mask[:] = 1
            masks.append(mask)
        return masks

    def encode_multiscale_feats(self, x, vessel_segs=None) -> List[torch.Tensor]:
        """
        Gets multiscale featuress from 3D UNET Encoder.
        """

        x = self.in_conv(x)
        x = self.in_dw(x)

        x1 = self.block1(x)
        x = self.block1_dw(x1)

        x2 = self.block2(x)
        x = self.block2_dw(x2)

        x3 = self.block3(x)
        x = self.block3_dw(x3)

        x4 = self.block4(x)

        feats = [x1, x2, x3, x4]

        multiscale_masks = None
        if vessel_segs is not None:
            segs = []
            for feat in feats:
                ups = torch.nn.Upsample(
                    size=(feat.shape[-3], feat.shape[-2], feat.shape[-1]),
                    # mode="nearest",
                    mode="trilinear",
                    # align_corners=True,
                )

                seg = ups(vessel_segs)
                seg = (seg >= 0.5).float()
                for b in range(seg.shape[0]):
                    if seg[b].sum() == 0:
                        # randomly chose one boxel to be 1, from dimensions 2,3,4
                        d = torch.randint(0, seg.shape[2], (1,))
                        h = torch.randint(0, seg.shape[3], (1,))
                        w = torch.randint(0, seg.shape[4], (1,))
                        seg[b, 0, d, h, w] = 1.0
                segs.append(seg)
            multiscale_masks = self.get_multiscale_mask(segs)

        return feats, multiscale_masks
