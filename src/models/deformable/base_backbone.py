"""
Base class for multiscale feature extraction backbones that
implements the forward method and the positional embedding
method
"""

import pdb
from typing import List, Tuple

import torch
from torch import nn

from src.utils.position_embedding import (
    create_global_pos_volume,
    get_3d_sinusoidal_pos_emb,
    get_3d_sinusoidal_pos_plus_vessel_emb,
)


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
        self.input_proj_list_v2: nn.ModuleList
        self.vessel_dist_proj_list: nn.ModuleList

    @property
    def device(self):
        """
        Generic method to get the device of the model.

        Probably won't work on model parallel setups?
        """
        return next(self.parameters()).device

    def encode_multiscale_feats(self, x, vessel_segs=None) -> List[torch.Tensor]:
        """
        Gets multiscale features from backbone.
        """
        raise NotImplementedError

    def get_positional_embeddings(
        self, multiscale_feats, vessel_dists=None, device="cpu"
    ):
        """
        Gets positional embeddings for every feature level.
        """
        factors = {
            0: 1,
            1: 2,
            2: 4,
            3: 8,
        }
        multiscale_pos_embs = []
        for lvl, feat in enumerate(multiscale_feats):
            pos_volume = create_global_pos_volume(*feat.shape[-3:]).to(device)
            if self.cfg.MODEL.DEFORMABLE.FIX_PE_SCALE:
                factor = factors[lvl]
                pos_volume = pos_volume * factor
            ups = torch.nn.Upsample(
                size=(feat.shape[-3], feat.shape[-2], feat.shape[-1]),
                mode="trilinear",
                # align_corners=True,
            )

            if vessel_dists is not None:

                vessel_dists_downsampled = ups(vessel_dists)
                pos_emb = get_3d_sinusoidal_pos_plus_vessel_emb(
                    pos_volume,
                    vessel_dists_downsampled,
                    self.src_patch_size,
                    num_pos_feats=self.output_hidden_dim // 4,
                    normalize=True,
                ).to(device)
            else:
                pos_emb = get_3d_sinusoidal_pos_emb(
                    pos_volume,
                    num_pos_feats=self.output_hidden_dim // 3,
                    normalize=True,
                ).to(device)

            multiscale_pos_embs.append(pos_emb)

        return multiscale_pos_embs

    def forward(self, x, vessel_dists=None, vessel_segs=None, level_emb=None) -> Tuple:
        """
        Encodes the input x and returns the multiscale features and positional
        embeddings.

        Parameters:
            x (torch.Tensor): input tensor of shape (N, C, D, H, W)

        Returns:
            Tuple: (multiscale_feats, multiscale_pos_embs)
        """
        multiscale_feats, multiscale_masks = self.encode_multiscale_feats(
            x, vessel_segs
        )
        #for feat in multiscale_feats:
        #    print(feat.shape)
        n_levels = len(multiscale_feats)
        # need to track a mask forf the global pos embedding with:
        # level-based indices
        # where to mask out non-vessel areas

        # worst csae scenario the mask has 1/384 of the memory size of the pos emb, or 2/384 if we keep two masks
        # if we keep these indices, there's no need to pass the pos emb thru the flatten_features, pad_flat_feats, and unpad_then_flatten_features functions

        multiscale_pos_embs = self.get_positional_embeddings(
            multiscale_feats,
            vessel_dists,
            # device="cpu" if self.cfg.MODEL.DEFORMABLE.EFFICIENT_MASK else self.device,
            device=self.device,
        )

        # print(multiscale_pos_embs[-1].shape, multiscale_feats[-1].shape, multiscale_masks[-1].shape)

        if vessel_segs is not None and self.cfg.MODEL.DEFORMABLE.EFFICIENT_MASK_V2:
            # V2: mask-first (like V1) but with LayerNorm instead of GroupNorm.
            # LayerNorm normalizes per-token independently, so zero-padded
            # positions cannot corrupt vessel token statistics.
            multiscale_feats, multiscale_masks, level_indices = self.flatten_features(
                multiscale_feats, multiscale_masks,
            )

            masked_pos_embs, masked_levels = self.mask_then_pad_levels_with_embs(
                multiscale_feats[0].shape[0],
                multiscale_pos_embs,
                level_indices,
                multiscale_masks,
            )
            masked_pos_embs = masked_pos_embs.to(self.device)

            multiscale_feats, multiscale_masks = self.pad_flat_feats(
                multiscale_feats, multiscale_masks
            )

            for i, feat in enumerate(multiscale_feats):
                multiscale_feats[i] = self.input_proj_list_v2[i](feat)

            multiscale_feats = [
                feat.squeeze(-1).squeeze(-1).transpose(1, 2)
                for feat in multiscale_feats
            ]
            multiscale_feats, multiscale_masks = self.unpad_then_flatten_features(
                multiscale_feats, multiscale_masks
            )

            for level in range(n_levels):
                lev_emb = level_emb[level]
                masked_pos_embs[masked_levels == level] += lev_emb.unsqueeze(0)
            multiscale_pos_embs = masked_pos_embs

        elif vessel_segs is not None and self.cfg.MODEL.DEFORMABLE.EFFICIENT_MASK_V3:
            # V3: project on full 3D volumes first (preserves GroupNorm stats),
            # then mask all levels in a single pass.
            # Gradient checkpointing wraps projection+masking together so the
            # large projected volumes [B, d_model, D, H, W] are intermediates
            # (not stored for backward) — only the small masked output is kept.
            n = len(multiscale_feats)
            all_args = (*multiscale_feats, *multiscale_pos_embs, *multiscale_masks, level_emb)
            multiscale_feats, multiscale_pos_embs, multiscale_masks = (
                torch.utils.checkpoint.checkpoint(
                    self._project_and_mask_all, n, *all_args, use_reentrant=False
                )
            )

        elif vessel_segs is not None and self.cfg.MODEL.DEFORMABLE.EFFICIENT_MASK:

            multiscale_feats, multiscale_masks, level_indices = self.flatten_features(
                multiscale_feats,
                multiscale_masks,
            )

            masked_pos_embs, masked_levels = self.mask_then_pad_levels_with_embs(
                multiscale_feats[0].shape[0],
                multiscale_pos_embs,
                level_indices,
                multiscale_masks,
            )
            masked_pos_embs = masked_pos_embs.to(self.device)

            multiscale_feats, multiscale_masks = self.pad_flat_feats(
                multiscale_feats, multiscale_masks
            )

            for i, feat in enumerate(multiscale_feats):
                multiscale_feats[i] = self.input_proj_list[i](feat)

            multiscale_feats = [
                feat.squeeze(-1).squeeze(-1).transpose(1, 2)
                for feat in multiscale_feats
            ]
            multiscale_feats, multiscale_masks = self.unpad_then_flatten_features(
                multiscale_feats, multiscale_masks
            )

            for level in range(n_levels):
                lev_emb = level_emb[level]
                masked_pos_embs[masked_levels == level] += lev_emb.unsqueeze(0)
            multiscale_pos_embs = masked_pos_embs

        else:
            for i, feat in enumerate(multiscale_feats):
                multiscale_feats[i] = self.input_proj_list[i](feat)
        # print(multiscale_feats.shape,  (~multiscale_masks).sum())
        return (multiscale_feats, multiscale_pos_embs, multiscale_masks)

    # def get_level_indices(self, multiscale_feats):
    #     level_indices = []
    #     for lvl, feat in enumerate(multiscale_feats):

    #         b, _, d, h, w = feat.shape
    #         level = torch.full((b, d * h * w), lvl, dtype=torch.long).to(feat.device)
    #         level_indices.append(level)
    #     return level_indices

    def _project_and_mask_all(self, n, *args):
        """
        Combined projection + masking, intended to be wrapped in
        torch.utils.checkpoint so the full projected volumes are
        intermediates (not stored for backward).
        """
        feats = list(args[:n])
        pos_embs = list(args[n : 2 * n])
        masks = list(args[2 * n : 3 * n])
        level_emb = args[3 * n]

        for i in range(n):
            feats[i] = self.input_proj_list[i](feats[i])

        return self.mask_flatten_all_levels(feats, pos_embs, masks, level_emb)

    def mask_flatten_all_levels(
        self, multiscale_feats, multiscale_pos_embs, multiscale_masks, level_emb
    ):
        """
        Flatten projected features from all levels, mask to vessel-only tokens,
        and pad across the batch — all in a single pass.

        Args:
            multiscale_feats: list of [B, d_model, D_i, H_i, W_i] (already projected)
            multiscale_pos_embs: list of [C, N_i] or [B, C, N_i] pos embeddings per level
            multiscale_masks: list of [B, 1, D_i, H_i, W_i] vessel masks per level
            level_emb: nn.Parameter [n_levels, d_model]

        Returns:
            (padded_feats, padded_pos_embs, attn_mask):
                padded_feats: [B, max_tokens, d_model]
                padded_pos_embs: [B, max_tokens, d_model]
                attn_mask: [B, max_tokens] (True = padding, for key_padding_mask)
        """
        bsz = multiscale_feats[0].shape[0]

        # Per batch element: collect vessel tokens from all levels
        batch_feats = []
        batch_pos_embs = []
        for b in range(bsz):
            b_feats = []
            b_pos = []
            for lvl, feat in enumerate(multiscale_feats):
                # Flatten spatial dims: [d_model, D, H, W] -> [N, d_model]
                feat_flat = feat[b].flatten(1).transpose(0, 1)  # [N_i, d_model]
                mask_flat = multiscale_masks[lvl][b].flatten().bool()  # [N_i]

                # Pos emb: [C, N_i] -> [N_i, C]
                pos_emb = multiscale_pos_embs[lvl]
                if pos_emb.dim() == 2:
                    pos_flat = pos_emb.transpose(0, 1)  # [N_i, C]
                else:  # [B, C, N_i]
                    pos_flat = pos_emb[b].transpose(0, 1)  # [N_i, C]

                # Add level embedding before masking
                lev_emb = level_emb[lvl]  # [d_model]
                pos_flat = pos_flat + lev_emb.unsqueeze(0)

                # Select vessel-only tokens
                b_feats.append(feat_flat[mask_flat.to(feat_flat.device)])
                b_pos.append(pos_flat[mask_flat.to(pos_flat.device)])

            batch_feats.append(torch.cat(b_feats, dim=0))
            batch_pos_embs.append(torch.cat(b_pos, dim=0))

        # Pad to max length across batch
        max_len = max(bf.shape[0] for bf in batch_feats)
        d_model = multiscale_feats[0].shape[1]
        device = multiscale_feats[0].device

        padded_feats = torch.zeros((bsz, max_len, d_model), device=device)
        padded_pos = torch.zeros((bsz, max_len, d_model), device=device)
        attn_mask = torch.zeros((bsz, max_len), device=device, dtype=torch.bool)

        for b in range(bsz):
            n = batch_feats[b].shape[0]
            padded_feats[b, :n] = batch_feats[b]
            padded_pos[b, :n] = batch_pos_embs[b]
            attn_mask[b, n:] = True  # mask out padding positions

        return padded_feats, padded_pos, attn_mask

    def mask_then_pad_levels_with_embs(
        self, bsz, multiscale_pos_embs, levels, multiscale_masks
    ):
        masked_pos_embs = []
        masked_levels = []
        for b in range(bsz):
            masked_pos_emb = []
            masked_level = []
            for lvl, pos_emb in enumerate(multiscale_pos_embs):
                pos_emb_reshaped = pos_emb.unsqueeze(0).transpose(1, 2)
                mask = multiscale_masks[lvl][b].bool().flatten(0)
                masked_pos_emb.append(
                    pos_emb_reshaped[0][mask.to(pos_emb_reshaped.device), :]
                )
                masked_level.append(levels[lvl][0][mask])
            masked_pos_embs.append(torch.cat(masked_pos_emb, dim=0))
            masked_levels.append(torch.cat(masked_level, dim=0))

        max_len = 0
        for mk in masked_pos_embs:
            if mk.shape[0] > max_len:
                max_len = mk.shape[0]
        padded_embs = torch.zeros(
            (len(masked_pos_embs), max_len, masked_pos_embs[0].shape[1]),
            device=masked_pos_embs[0].device,
        )
        padded_levels = torch.full(
            (len(masked_levels), max_len),
            -1,
            device=masked_pos_embs[0].device,
            dtype=torch.long,
        )
        for b, _ in enumerate(masked_pos_embs):
            me = masked_pos_embs[b]
            ml = masked_levels[b]
            padded_embs[b, : me.shape[0], :] = me
            padded_levels[b, : ml.shape[0]] = ml

        return padded_embs, padded_levels

    def unpad_then_flatten_features(self, multiscale_feats, multiscale_masks):
        bsz = multiscale_feats[0].shape[0]
        feats_flatten = []
        for b in range(bsz):
            feats_batch = []
            for lvl, _ in enumerate(multiscale_feats):
                mask = multiscale_masks[lvl][b].bool()
                feats_batch.append(multiscale_feats[lvl][b][mask, :])
            feats_flatten.append(torch.cat(feats_batch, dim=0))
        # pad the flattened features to the max length in the batch
        max_length = 0
        for mk in feats_flatten:
            if mk.shape[0] > max_length:
                max_length = mk.shape[0]

        padded_feats = torch.zeros(
            (bsz, max_length, multiscale_feats[0].shape[2]),
            device=multiscale_feats[0].device,
        )
        attn_mask = torch.zeros(
            (bsz, max_length), device=multiscale_feats[0].device, dtype=torch.bool
        )
        for b in range(bsz):
            mf = feats_flatten[b]
            padded_feats[b, : mf.shape[0], :] = mf
            attn_mask[b, mf.shape[0] :] = True
        return padded_feats, attn_mask

    def pad_flat_feats(self, flatten_feats, flatten_masks):

        padded_feats = []
        padded_masks = []
        for feats, masks in zip(flatten_feats, flatten_masks):
            masked_feat = []

            for b in range(feats.shape[0]):
                mask = masks[b].bool()
                masked_feat.append(feats[b][mask])
            max_len = max([mf.shape[0] for mf in masked_feat])
            # create an empty tensor to put the padded features in
            padded_feat = torch.zeros(
                (len(masked_feat), max_len, feats.shape[2]), device=feats[0].device
            )

            padded_mask = torch.zeros(
                (len(masks), max_len), device=masks[0].device, dtype=torch.bool
            )

            for b, _ in enumerate(masked_feat):
                cur_len = masked_feat[b].shape[0]
                padded_feat[b, :cur_len, :] = masked_feat[b]
                padded_mask[b, :cur_len] = 1
            # reorder padded_feat to be B, C, D and then unsqueeze to have tow
            # extra dimensions at the end
            padded_feat = padded_feat.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
            padded_feats.append(padded_feat)
            padded_masks.append(padded_mask)
        return padded_feats, padded_masks

    def flatten_features(self, multiscale_feats, multiscale_masks):
        bs = multiscale_feats[0].shape[0]

        feats_flatten = []
        spatial_shapes = []
        masks_flatten = []
        level_indices = []

        for (
            lvl,
            feat,
        ) in enumerate(multiscale_feats):

            b, _, d, h, w = feat.shape
            level = torch.full((b, d * h * w), lvl, dtype=torch.long).to(feat.device)
            level_indices.append(level)
            spatial_shape = d, h, w
            spatial_shapes.append(spatial_shape)
            # flatten feat from B C D H W into B DHW C
            feats_flatten.append(feat.flatten(2).permute(0, 2, 1))

            if multiscale_masks is not None:
                mask = multiscale_masks[lvl]
                masks_flatten.append(mask.flatten(1))

        return feats_flatten, masks_flatten, level_indices

        # if self.decoder_only:
        #     global_feats = feats_flatten
        # else:
        #     global_feats = self.encoder(
        #         feats_flatten,
        #         spatial_shapes,
        #         level_start_index,
        #         pos_embed_flatten,
        #     )            )
