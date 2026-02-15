"""
Conditional DETR Decoder for the efficient transformer branch.

Implements the Conditional DETR cross-attention decomposition (Meng et al., ICCV 2021)
adapted for 3D detection with sparse features. The spatial query is derived from
sinusoidal(ref_loc) gated by FFN(decoder_output), replacing the standard learned
positional embedding in cross-attention.

Self-attention remains standard (additive positional embeddings).
"""

import torch
import torch.nn.functional as F
from torch import nn

from src.models.layers.generic_mlp import GenericMLP
from src.utils.general import get_activation_fn, get_clones, inverse_sigmoid
from src.utils.position_embedding import get_3d_sinusoidal_pos_emb


class ConditionalCrossAttention(nn.Module):
    """
    Cross-attention for Conditional DETR with per-head content/spatial concatenation.

    Q and K arrive pre-concatenated at 2*d_model (content + spatial interleaved per head).
    V is content-only at d_model. Uses SDPA for flash attention.
    """

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_model // n_heads
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout

    def forward(self, q, k, v, key_padding_mask=None):
        """
        Parameters
        ----------
        q : [bs, n_q, 2*d_model] per-head concatenated content + spatial
        k : [bs, n_k, 2*d_model] per-head concatenated content + spatial
        v : [bs, n_k, d_model] content features
        key_padding_mask : [bs, n_k] True = padding (masked out)
        """
        bs, n_q, _ = q.shape
        n_k = k.shape[1]

        v = self.v_proj(v)

        q = q.view(bs, n_q, self.n_heads, 2 * self.d_head).transpose(1, 2)
        k = k.view(bs, n_k, self.n_heads, 2 * self.d_head).transpose(1, 2)
        v = v.view(bs, n_k, self.n_heads, self.d_head).transpose(1, 2)

        attn_mask = None
        if key_padding_mask is not None:
            # Convert key_padding_mask (True=padding) to float mask (-inf=masked)
            attn_mask = torch.zeros(
                bs, 1, 1, n_k, dtype=q.dtype, device=q.device
            )
            attn_mask.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )  # [bs, n_heads, n_q, d_head]

        out = out.transpose(1, 2).contiguous().view(bs, n_q, self.d_model)
        return self.out_proj(out)


def _sinusoidal_pos_emb_batched(pos, num_pos_feats, temperature=10000):
    """
    Compute 3D sinusoidal positional embedding for batched input.

    Parameters
    ----------
    pos : torch.Tensor
        [bs, N, 3] coordinates (expected in [0, 1] range).
    num_pos_feats : int
        Number of features per spatial dimension.

    Returns
    -------
    torch.Tensor
        [bs, N, num_pos_feats * 3] sinusoidal positional embeddings.
    """
    bs, n, _ = pos.shape
    # get_3d_sinusoidal_pos_emb expects [N, 3] and returns [d_model, N]
    # Process per sample to handle the permute(1, 0) inside the function.
    embeds = []
    for i in range(bs):
        emb = get_3d_sinusoidal_pos_emb(
            pos[i], num_pos_feats=num_pos_feats, temperature=temperature, normalize=False
        )  # [d_model, N]
        embeds.append(emb.transpose(0, 1))  # [N, d_model]
    return torch.stack(embeds, dim=0)  # [bs, N, d_model]


class ConditionalTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_heads=8,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # self attention (unchanged from standard decoder)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # content projections for cross attention (applied before per-head concat)
        self.q_content_proj = nn.Linear(d_model, d_model)
        self.k_content_proj = nn.Linear(d_model, d_model)

        # spatial projections for cross attention (applied before per-head concat)
        self.q_pos_sine_proj = nn.Linear(d_model, d_model)
        self.k_pos_proj = nn.Linear(d_model, d_model)

        # first-layer additive learned PE for cross-attn query (set to None for layers 1+)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)

        # conditional cross attention (concatenated content + spatial, SDPA)
        self.cross_attn = ConditionalCrossAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # spatial gate FFN: decoder content -> gating vector for spatial query
        self.spatial_gate_ffn = GenericMLP(
            input_dim=d_model,
            hidden_dims=[d_model],
            output_dim=d_model,
            norm_fn_name="ln",
            activation="relu",
            use_conv=False,
        )

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(x, pos):
        return x if pos is None else x + pos

    def forward_ffn(self, x):
        x2 = self.linear2(self.dropout3(self.activation(self.linear1(x))))
        x = x + self.dropout4(x2)
        x = self.norm3(x)
        return x

    def forward(
        self,
        ref,
        ref_loc,
        ref_pos_embed,
        global_feats,
        global_pos_embed,
        layer_idx,
        key_padding_mask=None,
    ):
        # self attention (unchanged: additive positional embeddings)
        q = k = self.with_pos_embed(ref, ref_pos_embed)
        ref2 = self.self_attn(
            q.transpose(0, 1), k.transpose(0, 1), ref.transpose(0, 1)
        )[0].transpose(0, 1)
        ref = ref + self.dropout2(ref2)
        ref = self.norm2(ref)

        # conditional cross attention
        # Spatial query: sinusoidal(ref_loc) * gate(decoder_content)
        num_pos_feats = self.d_model // 3
        p_s = _sinusoidal_pos_emb_batched(
            ref_loc, num_pos_feats=num_pos_feats
        )  # [bs, n_q, d_model]

        if layer_idx == 0:
            p_q = p_s
        else:
            lambda_q = self.spatial_gate_ffn(ref)  # [bs, n_q, d_model]
            p_q = p_s * lambda_q

        bs, n_q, _ = ref.shape
        n_k = global_feats.shape[1]
        d_head = self.d_model // self.n_heads

        # Project content and spatial streams before per-head concatenation
        content_q = self.q_content_proj(ref)
        content_k = self.k_content_proj(global_feats)
        spatial_q = self.q_pos_sine_proj(p_q)
        spatial_k = self.k_pos_proj(global_pos_embed)

        # First layer: add projected learned PE to content query (like original Cond. DETR)
        if self.ca_qpos_proj is not None:
            content_q = content_q + self.ca_qpos_proj(ref_pos_embed)

        # Per-head concatenation: [bs, n, d_model] -> [bs, n, nhead, d_head] -> cat dim=3 -> [bs, n, nhead, 2*d_head] -> [bs, n, 2*d_model]
        q = torch.cat([
            content_q.view(bs, n_q, self.n_heads, d_head),
            spatial_q.view(bs, n_q, self.n_heads, d_head),
        ], dim=3).view(bs, n_q, self.d_model * 2)
        k = torch.cat([
            content_k.view(bs, n_k, self.n_heads, d_head),
            spatial_k.view(bs, n_k, self.n_heads, d_head),
        ], dim=3).view(bs, n_k, self.d_model * 2)

        ref2 = self.cross_attn(q, k, global_feats, key_padding_mask=key_padding_mask)

        ref = ref + self.dropout1(ref2)
        ref = self.norm1(ref)

        # ffn
        ref = self.forward_ffn(ref)

        return ref


class ConditionalTransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers,
        with_recurrence=False,
        with_stepwise_loss=False,
        shared_heads=False,
        return_intermediate=False,
    ):
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        # Nullify ca_qpos_proj for layers 1+ (only layer 0 uses additive learned PE)
        for layer_id in range(num_layers - 1):
            self.layers[layer_id + 1].ca_qpos_proj = None
        self.return_intermediate = return_intermediate
        self.with_recurrence = with_recurrence
        self.with_stepwise_loss = with_stepwise_loss
        self.bbox_embed = None
        self.class_embed = None
        self.shared_heads = shared_heads
        self.center_head: torch.nn.Module
        self.class_head: torch.nn.Module
        self.size_head: torch.nn.Module

    def select_center_head(self, layer):
        if self.shared_heads:
            return self.center_head
        else:
            return self.center_head[layer]  # type: ignore

    def select_class_head(self, layer):
        if self.shared_heads:
            return self.class_head
        else:
            return self.class_head[layer]  # type: ignore

    def select_size_head(self, layer):
        if self.shared_heads:
            return self.size_head
        else:
            return self.size_head[layer]  # type: ignore

    def refine_center(self, lid, reference_points, output):
        tmp = self.select_center_head(lid)(output.permute(0, 2, 1)).permute(0, 2, 1)
        new_reference_points = tmp + inverse_sigmoid(reference_points)
        new_reference_points = new_reference_points.sigmoid()
        return new_reference_points

    def forward(
        self,
        query,
        ref_loc,
        query_pos_embed,
        global_feats,
        global_pos_embed,
        key_padding_mask=None,
    ):
        output = query

        intermediate = []
        prev_ref_loc = ref_loc
        intermediate_ref_locs = []
        box_prediction_list = []
        viz_outputs_list = []
        for lid, layer in enumerate(self.layers):
            output = layer(
                output,
                ref_loc,
                query_pos_embed,
                global_feats,
                global_pos_embed,
                layer_idx=lid,
                key_padding_mask=key_padding_mask,
            )

            viz_outputs_list.append({})

            if lid == self.num_layers - 1 or self.with_stepwise_loss:
                prev_ref_loc = ref_loc
                ref_loc = self.refine_center(lid, ref_loc, output)
                class_logits = self.select_class_head(lid)(
                    output.permute(0, 2, 1)
                ).permute(0, 2, 1)
                size = (
                    self.select_size_head(lid)(output.permute(0, 2, 1))
                    .permute(0, 2, 1)
                    .sigmoid()
                )
                class_probs = torch.nn.functional.softmax(class_logits, dim=-1)

                box_dict = {
                    "class_logits": class_logits,
                    "size": size,
                    "center": ref_loc,
                    "pre_refinement_center": prev_ref_loc,
                    "class_probs": class_probs,
                }
                box_prediction_list.append(box_dict)

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_ref_locs.append(ref_loc)

        if self.return_intermediate:
            ref_loc = torch.stack(intermediate_ref_locs)

        return box_prediction_list, viz_outputs_list
