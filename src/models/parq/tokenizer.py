# Copyright (c) Meta Platforms, Inc. and affiliates.
import logging
from typing import Optional

import einops
import torch


class ImageToPatchTokens(torch.nn.Module):
    """
    Patchify an image or image feature (Image grid --> sequence of patches)
    """

    def __init__(self, patch_size: tuple):
        super(ImageToPatchTokens, self).__init__()
        self.patch_size = patch_size

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        input:
            img:     tensor, (B, C, T, H, W), A snippet of images or image features
        output:
            patches: tensor, (B, T/P, H/P, W/P, P*P*P*C), P is the patch size
        """
        patches = einops.rearrange(
            img,
            "b c (t dt) (h dh) (w dw) -> b t h w (dt dh dw c)",
            dh=self.patch_size[1],
            dw=self.patch_size[2],
            dt=self.patch_size[0],
        )
        return patches


class ImageSeqTokenizer(torch.nn.Module):
    """
    Tokenize a volume for the Transformer by:
        1. Convert to sequence of patches
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
    ):
        """
        Args:
            in_channels:  Input number of channels in the feature volume
            out_channels: Output channels required from model
            patch_size:   size of patch to divide image in
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size

        self.to_tokens = ImageToPatchTokens(patch_size=self.patch_size)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_channels, out_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels, out_channels),
        )
        patch_encoding_out = self.out_channels

        # self.token_position_encoder = ImagePositionEncoding(
        #     dim_out=patch_encoding_out,
        #     ray_points_scale=ray_points_scale,
        #     num_samples=num_samples,
        #     min_depth=min_depth,
        #     max_depth=max_depth,
        # )

        for m in self.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

    def forward(
        self,
        volume: torch.Tensor,
    ) -> torch.Tensor:
        """
        input:
            volume:             (B, T, C, H, W), image features
        output:
            ret:                (B, T*H*W, C), tokenized image features
        """
        assert volume.dim() == 5, f"Images needs to have 5 dimensions {volume.shape}"
        # token_pos_enc = self.token_position_encoder(
        #     B=volume.shape[0],
        #     T=volume.shape[1],
        #     camera=camera,
        #     T_camera_pseudoCam=T_camera_pseudoCam,
        #     T_world_pseudoCam=T_world_pseudoCam,
        #     T_local_world=T_world_local.inverse(),
        # )

        # ret = volume + token_pos_enc
        ret = self.to_tokens(volume)
        ret = einops.rearrange(ret, "b t h w c -> b t (h w) c")
        ret = einops.rearrange(ret, "b t n c -> b (t n) c")
        ret = self.encoder(ret)
        return ret
