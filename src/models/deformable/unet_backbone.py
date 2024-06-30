from typing import List

import torch
from src.models.deformable.base_backbone import Base_Backbone
from src.models.layers.conv_layers import (
    ConvBlock,
    DownsamplingConvBlock,
    LayerBasic,
    UpsamplingDeconvBlock,
)
from torch import nn

# Inherit from CNN_Encoder


def build_backbone(cfg):
    """
    Builds the UNET Encoder for the PARQ model.
    """
    return UNET_Backbone(
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


class UNET_Backbone(Base_Backbone):
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
        super(UNET_Backbone, self).__init__(cfg)
        assert n_levels >= 1 and n_levels <= 4
        # TODO: make this matter
        self.cfg = cfg
        self.output_hidden_dim = output_hidden_dim
        self.n_levels = n_levels

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

        self.block33_up = UpsamplingDeconvBlock(
            n_filters[3], n_filters[2], norm_type=norm_type, act_type=act_type
        )
        self.block33_res = LayerBasic(
            1, n_filters[2], n_filters[2], norm_type=norm_type, act_type=act_type, se=se
        )
        self.block33 = LayerBasic(
            2,
            n_filters[2] * 2,
            n_filters[2],
            norm_type=norm_type,
            act_type=act_type,
            se=se,
        )

        self.block22_up = UpsamplingDeconvBlock(
            n_filters[2], n_filters[1], norm_type=norm_type, act_type=act_type
        )
        self.block22_res = LayerBasic(
            1, n_filters[1], n_filters[1], norm_type=norm_type, act_type=act_type, se=se
        )
        self.block22 = LayerBasic(
            2,
            n_filters[1] * 2,
            n_filters[1],
            norm_type=norm_type,
            act_type=act_type,
            se=se,
        )
        self.layer_hidden_dims = [
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

    def encode_multiscale_feats(self, x) -> List[torch.Tensor]:
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

        x_bottleneck = self.block4(x)

        # "decode"
        x = self.block33_up(x_bottleneck)
        x3 = self.block33_res(x3)

        x = torch.cat([x, x3], dim=1)
        x_out_first_upsample = x = self.block33(x)

        x = self.block22_up(x)
        x2 = self.block22_res(x2)

        x = torch.cat([x, x2], dim=1)
        x_out_second_upsample = x = self.block22(x)
        return [x_out_second_upsample, x_out_first_upsample, x_bottleneck]
