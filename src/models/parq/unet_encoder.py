from collections import OrderedDict

import torch
from src.models.layers.conv_layers import (
    ConvBlock,
    DownsamplingConvBlock,
    LayerBasic,
    UpsamplingDeconvBlock,
)
from torch import nn


class UNET_Encoder(nn.Module):
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
        device=None,
        use_pretrained_encoder=False,
        path_unet_weights="",
        frozen_pretrained_encoder: bool = False,
    ):
        super(UNET_Encoder, self).__init__()
        self.cfg = cfg

        self.device = device
        self._split_comb = None

        # pretrained UNET
        self.use_pretrained_encoder = use_pretrained_encoder
        self.path_unet_weights = path_unet_weights
        self.frozen_pretrained_encoder = frozen_pretrained_encoder
        self.frozen_parameters_list = []

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

        if self.use_pretrained_encoder and self.path_unet_weights != "":
            self.load_pretrained_encoder_weights()

            if self.frozen_pretrained_encoder:
                self.freeze_parameters()

    def forward(self, x):

        # "encode"
        x = self.in_conv(x)
        x = self.in_dw(x)

        x1 = self.block1(x)
        x = self.block1_dw(x1)

        x2 = self.block2(x)
        x = self.block2_dw(x2)

        x3 = self.block3(x)
        x = self.block3_dw(x3)

        x = self.block4(x)

        # "decode"
        x = self.block33_up(x)
        x3 = self.block33_res(x3)

        x = torch.cat([x, x3], dim=1)
        x = self.block33(x)

        x = self.block22_up(x)
        x2 = self.block22_res(x2)

        x = torch.cat([x, x2], dim=1)
        x = self.block22(x)
        return x

    def freeze_parameters(self):
        """
        Freezes the parameters defined in `self.frozen_parameters_list`.
        """
        for name, param in self.named_parameters():
            if name in self.frozen_parameters_list:
                param.requires_grad = False
        print("\nParameters frozen successfully")
        return

    def load_pretrained_encoder_weights(self):
        """
        Loads weights from a pretrained UNET model into the encoder
        to make the training more stable.
        """
        unet_weights = torch.load(
            self.path_unet_weights, map_location=torch.device("cpu")
        )["model"]
        unet_keys = list(unet_weights.keys())
        remapped_unet_keys = {
            unet_key.replace("model.", ""): unet_weights[unet_key]
            for unet_key in unet_keys
        }
        remapped_unet_keys = OrderedDict(remapped_unet_keys)
        model_keys = list(self.state_dict().keys())
        # remove keys not in model_keys
        remapped_unet_keys = {
            model_key: remapped_unet_keys[model_key]
            for model_key in model_keys
            if model_key in remapped_unet_keys
        }
        list_unet_keys = list(remapped_unet_keys.keys())
        self.frozen_parameters_list.extend(list_unet_keys)
        self.load_state_dict(remapped_unet_keys, strict=True)
        print(
            "\nLoaded pretrained UNET weights"
            + f"from {self.path_unet_weights} successfully"
        )
        return
