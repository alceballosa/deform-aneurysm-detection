import torch
from torch import nn

from src.utils.general import get_activation_fn, get_clones


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
    ):
        super().__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(x, pos):
        return x if pos is None else x + pos

    def forward_ffn(self, x):
        x2 = self.linear2(self.dropout2(self.activation(self.linear1(x))))
        x = x + self.dropout3(x2)
        x = self.norm2(x)
        return x

    def forward(self, global_feats, global_pos_embed, key_padding_mask):
        # self attention
        q = k = v = self.with_pos_embed(global_feats, global_pos_embed)
        global_feats2, _ = self.self_attn(
            query=q,
            key=k,
            value=v,
            key_padding_mask=key_padding_mask,
        )
        global_feats = global_feats + self.dropout1(global_feats2)
        global_feats = self.norm1(global_feats)
        # ffn
        global_feats = self.forward_ffn(global_feats)
        return global_feats


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, use_checkpoint=False):
        super().__init__()
        self.layers = get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.use_checkpoint = use_checkpoint

    def forward(
        self,
        global_feats,
        global_pos_embed,
        key_padding_mask=None,
    ):

        output = global_feats
        for _, layer in enumerate(self.layers):
            if self.use_checkpoint and self.training:
                # Use gradient checkpointing to save memory during training
                # Trades compute (recomputation) for memory (storing activations)
                output = torch.utils.checkpoint.checkpoint(
                    layer,
                    output,
                    global_pos_embed,
                    key_padding_mask,
                    use_reentrant=False,
                )
            else:
                output = layer(output, global_pos_embed, key_padding_mask)

        return output
