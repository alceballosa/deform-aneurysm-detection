"""
Base class for multiscale feature extraction backbones that
implements the forward method and the positional embedding
method
"""

from typing import List, Tuple

import torch
import torchio as tio
from src.utils.position_embedding import (
    create_global_pos_volume,
    get_3d_sinusoidal_pos_emb,
    get_3d_sinusoidal_pos_plus_vessel_emb,
)
from torch import nn


class Base_Backbone(nn.Module):
    """
    UNET-based encoder for the PARQ model.
    """

    def __init__(
        self,
        cfg,
    ):
        super(Base_Backbone, self).__init__()
        self.cfg = cfg
        self.src_patch_size = cfg.DATA.PATCH_SIZE
        self.output_hidden_dim: int
        self.input_proj_list: nn.ModuleList
        self.vessel_dist_proj_list: nn.ModuleList

    @property
    def device(self):
        """
        Generic method to get the device of the model.

        Probably won't work on model parallel setups?
        """
        return next(self.parameters()).device

    def encode_multiscale_feats(self, x) -> List[torch.Tensor]:
        """
        Gets multiscale features from backbone.
        """
        raise NotImplementedError

    def get_positional_embeddings(self, multiscale_feats, vessel_dists=None):
        """
        Gets positional embeddings for every feature level.
        """
        multiscale_pos_embs = []
        for feat in multiscale_feats:
            pos_volume = create_global_pos_volume(*feat.shape[-3:]).to(self.device)
            if vessel_dists is not None:
                # resize vessel_dists to match the eature level
                transform = tio.transforms.Resize(
                    (feat.shape[-3], feat.shape[-2], feat.shape[-1])
                )
                vessel_dists_downsampled = torch.stack(
                    [
                        transform(vessel_dists[i].cpu())
                        for i in range(vessel_dists.shape[0])
                    ],  # type: ignore
                    dim=0,
                ).to(self.device)
                pos_emb = get_3d_sinusoidal_pos_plus_vessel_emb(
                    pos_volume,
                    vessel_dists_downsampled,
                    self.src_patch_size,
                    num_pos_feats=self.output_hidden_dim // 4,
                    normalize=True,
                ).to(self.device)
            else:
                pos_emb = get_3d_sinusoidal_pos_emb(
                    pos_volume,
                    num_pos_feats=self.output_hidden_dim // 3,
                    normalize=True,
                ).to(self.device)

            multiscale_pos_embs.append(pos_emb)

        return multiscale_pos_embs

    def forward(self, x, vessel_dists=None) -> Tuple:
        """
        Encodes the input x and returns the multiscale features and positional
        embeddings.

        Parameters:
            x (torch.Tensor): input tensor of shape (N, C, D, H, W)

        Returns:
            Tuple: (multiscale_feats, multiscale_pos_embs)
        """
        multiscale_feats = self.encode_multiscale_feats(x)
        multiscale_pos_embs = self.get_positional_embeddings(
            multiscale_feats, vessel_dists
        )
        for i, feat in enumerate(multiscale_feats):
            multiscale_feats[i] = self.input_proj_list[i](feat)
        return (multiscale_feats, multiscale_pos_embs)
