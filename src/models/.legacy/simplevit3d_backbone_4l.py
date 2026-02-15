# From https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vivit.py

from typing import List

import numpy as np
import torch
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from src.models.deformable.base_backbone import Base_Backbone
from torch import nn
from torch.nn import functional as F

# import F from torch


# helpers


def build_backbone(cfg):
    """
    Builds the UNET Encoder for the PARQ model.
    """

    target_spatial_shape = 8
    image_size = cfg.DATA.PATCH_SIZE[1]

    trx_patch_size = image_size // target_spatial_shape

    v = SimpleViT(
        cfg=cfg,
        image_size=cfg.DATA.PATCH_SIZE[1],  # image size
        frames=cfg.DATA.PATCH_SIZE[0],  # number of frames
        image_patch_size=trx_patch_size,  # image patch size
        frame_patch_size=trx_patch_size,  # frame patch size
        depth=3,
        num_classes=1,
        channels=cfg.DATA.N_CHANNELS,
        dim=cfg.MODEL.D_MODEL,
        heads=8,
        mlp_dim=cfg.MODEL.D_MODEL * 2,
    )
    return v


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def identity(x):
    return x


def posemb_sincos_3d(patches, temperature=10000, dtype=torch.float32):
    _, f, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    z, y, x = torch.meshgrid(
        torch.arange(f, device=device),
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )

    fourier_dim = dim // 6

    omega = torch.arange(fourier_dim, device=device) / (fourier_dim - 1)
    omega = 1.0 / (temperature**omega)

    z = z.flatten()[:, None] * omega[None, :]
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]

    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos(), z.sin(), z.cos()), dim=1)

    pe = F.pad(
        pe, (0, dim - (fourier_dim * 6))
    )  # pad if feature dimension not cleanly divisible by 6
    return pe.type(dtype)


# classes
class Conv3d(torch.nn.Conv3d):
    """
    A wrapper around :class:`torch.nn.Conv3d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv3d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.

        x = F.conv3d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x




class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head),
                        FeedForward(dim, mlp_dim),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class SimpleViT(Base_Backbone):
    def __init__(
        self,
        *,
        cfg,
        image_size,
        image_patch_size,
        frames,
        frame_patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        channels=3,
        dim_head=64
    ):
        super(SimpleViT, self).__init__(cfg)
        self.input_proj_list = [identity, identity, identity, identity]

        self.output_hidden_dim = dim
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."
        assert (
            frames % frame_patch_size == 0
        ), "Frames must be divisible by the frame patch size"

        num_patches = (
            (image_height // patch_height)
            * (image_width // patch_width)
            * (frames // frame_patch_size)
        )
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (f pf) (h p1) (w p2) -> b f h w (p1 p2 pf c)",
                p1=patch_height,
                p2=patch_width,
                pf=frame_patch_size,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.scale_factors = (4.0, 2.0, 1.0, 0.5)
        self.stages = []
        norm = "LN"
        use_bias = norm == ""
        spatial_scales = [8 * 4, 8 * 2, 8 * 1, 8 // 2]

        for idx, scale in enumerate(self.scale_factors):
            # TODO: verify this
            spatial_scale = spatial_scales[idx]
            out_channels = self.output_hidden_dim
            out_dim = dim

            if scale == 4.0:
                layers = [
                    nn.ConvTranspose3d(dim, dim // 2, kernel_size=2, stride=2),
                    get_norm(
                        norm,
                        [
                            dim // 2,
                            spatial_scales[1],
                            spatial_scales[1],
                            spatial_scales[1],
                        ],
                    ),
                    nn.GELU(),
                    nn.ConvTranspose3d(dim // 2, dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose3d(dim, dim // 2, kernel_size=2, stride=2)]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool3d(kernel_size=2, stride=2)]

            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")
            layers.extend(
                [
                    Conv3d(
                        out_dim,
                        out_channels,
                        kernel_size=1,
                        bias=use_bias,
                        norm=get_norm(
                            norm,
                            [out_channels, spatial_scale, spatial_scale, spatial_scale],
                        ),
                    ),
                    Conv3d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=use_bias,
                        norm=get_norm(
                            norm,
                            [out_channels, spatial_scale, spatial_scale, spatial_scale],
                        ),
                    ),
                ]
            )
            layers = nn.Sequential(*layers)
            self.stages.append(layers)
        self.stages = nn.ModuleList(self.stages)


    def forward_vit(self, video):
        *_, h, w, dtype = *video.shape, video.dtype

        x = self.to_patch_embedding(video)
        pe = posemb_sincos_3d(x)
        x = rearrange(x, "b ... d -> b (...) d") + pe

        x = self.transformer(x)
        # x = x.mean(dim = 1)
        spatial_dim = int(np.cbrt(x.shape[-2]))

        x = rearrange(
            x, "b (z x y) c -> b c z x y", x=spatial_dim, y=spatial_dim, z=spatial_dim
        ).contiguous()
        # x = self.to_latent(x)
        return x  # self.linear_head(x)


    def encode_multiscale_feats(self, x) -> List:
        x = self.forward_vit(x)
        results = []
        for i, stage in enumerate(self.stages):
            x_stage = stage(x)
            results.append(x_stage)

        return results

