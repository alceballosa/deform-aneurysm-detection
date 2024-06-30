import torch
import torch.nn as nn
import torch.nn.functional as F


def activation(act="ReLU"):
    if act == "ReLU":
        return nn.ReLU(inplace=True)
    elif act == "LeakyReLU":
        return nn.LeakyReLU(inplace=True)
    elif act == "ELU":
        return nn.ELU(inplace=True)
    elif act == "PReLU":
        return nn.PReLU(inplace=True)
    else:
        return nn.Identity()


def norm_layer3d(norm_type, num_features):
    norm = {
        "BN": nn.BatchNorm3d,
        "GN": lambda channels: nn.GroupNorm(
            num_groups=channels // 8, num_channels=channels
        ),
        "SyncBN": nn.SyncBatchNorm,
        "LN": LayerNorm,
        "IN": lambda channels: nn.InstanceNorm3d(num_features=channels, affine=True),
        "": nn.Identity,
        "none": nn.Identity,
    }[norm_type](num_features)
    return norm


class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        stride=1,
        groups=1,
        norm_type="none",
        act_type="ReLU",
    ):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=kernel_size // 2 + dilation - 1,
            dilation=dilation,
            bias=False,
        )
        self.norm = norm_layer3d(norm_type, out_channels)
        self.act = activation(act_type)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class BasicBlockNew(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        norm_type="BN",
        act_type="ReLU",
        se=True,
    ):
        super(BasicBlockNew, self).__init__()

        self.conv1 = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            act_type=act_type,
            norm_type=norm_type,
        )

        self.conv2 = ConvBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            stride=1,
            act_type="none",
            norm_type=norm_type,
        )

        if in_channels == out_channels and stride == 1:
            self.res = nn.Identity()
        elif in_channels != out_channels and stride == 1:
            self.res = ConvBlock(
                in_channels,
                out_channels,
                kernel_size=1,
                act_type="none",
                norm_type=norm_type,
            )
        elif in_channels != out_channels and stride > 1:
            self.res = nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                ConvBlock(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    act_type="none",
                    norm_type=norm_type,
                ),
            )

        if se:
            self.se = SELayer(out_channels)
        else:
            self.se = nn.Identity()

        self.act = activation(act_type)

    def forward(self, x):
        ident = self.res(x)

        x = self.conv1(x)
        x = self.conv2(x)

        x = self.se(x)

        x += ident
        x = self.act(x)

        return x


class LayerBasic(nn.Module):
    def __init__(
        self,
        n_stages,
        in_channels,
        out_channels,
        stride=1,
        norm_type="BN",
        act_type="ReLU",
        se=False,
    ):
        super(LayerBasic, self).__init__()
        self.n_stages = n_stages
        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = in_channels
                stride = stride
            else:
                input_channel = out_channels
                stride = 1

            ops.append(
                BasicBlockNew(
                    input_channel,
                    out_channels,
                    stride=stride,
                    norm_type=norm_type,
                    act_type=act_type,
                    se=se,
                )
            )

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_type="BN",
        act_type="ReLU",
    ):
        super(DownsamplingConvBlock, self).__init__()

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=2,
            padding=0,
            stride=stride,
            bias=False,
        )
        self.norm = norm_layer3d(norm_type, out_channels)
        self.act = activation(act_type)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class DownsamplingBlock(nn.Module):
    def __init__(
        self,
        in_channels=None,
        out_channels=None,
        stride=2,
        pool_type="max",
        norm_type="BN",
        act_type="ReLU",
    ):
        super(DownsamplingBlock, self).__init__()

        if pool_type == "avg":
            self.down = nn.AvgPool3d(kernel_size=stride, stride=stride)
        else:
            self.down = nn.MaxPool3d(kernel_size=stride, stride=stride)
        if (in_channels is not None) and (out_channels is not None):
            self.conv = ConvBlock(
                in_channels, out_channels, 1, norm_type=norm_type, act_type=act_type
            )

    def forward(self, x):
        x = self.down(x)
        if hasattr(self, "conv"):
            x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_type="BN",
        act_type="ReLU",
    ):
        super(UpsamplingDeconvBlock, self).__init__()

        self.conv = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=stride,
            padding=0,
            stride=stride,
            bias=False,
        )
        self.norm = norm_layer3d(norm_type, out_channels)
        self.act = activation(act_type)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class UpsamplingBlock(nn.Module):
    def __init__(
        self,
        in_channels=None,
        out_channels=None,
        stride=2,
        mode="nearest",
        norm_type="BN",
        act_type="ReLU",
    ):
        super(UpsamplingBlock, self).__init__()

        self.up = nn.Upsample(scale_factor=stride, mode=mode)
        if (in_channels is not None) and (out_channels is not None):
            self.conv = ConvBlock(
                in_channels, out_channels, 1, norm_type=norm_type, act_type=act_type
            )

    def forward(self, x):
        if hasattr(self, "conv"):
            x = self.conv(x)
        x = self.up(x)
        return x


class ASPP(nn.Module):
    def __init__(
        self,
        channels,
        ratio=4,
        dilations=[1, 2, 3, 4],
        norm_type="BN",
        act_type="ReLU",
    ):
        super(ASPP, self).__init__()
        # assert dilations[0] == 1, 'The first item in dilations should be `1`'
        inner_channels = channels // ratio
        cat_channels = inner_channels * 5
        self.aspp0 = ConvBlock(
            channels,
            inner_channels,
            kernel_size=1,
            dilation=dilations[0],
            norm_type=norm_type,
            act_type=act_type,
        )
        self.aspp1 = ConvBlock(
            channels,
            inner_channels,
            kernel_size=3,
            dilation=dilations[1],
            norm_type=norm_type,
            act_type=act_type,
        )
        self.aspp2 = ConvBlock(
            channels,
            inner_channels,
            kernel_size=3,
            dilation=dilations[2],
            norm_type=norm_type,
            act_type=act_type,
        )
        self.aspp3 = ConvBlock(
            channels,
            inner_channels,
            kernel_size=3,
            dilation=dilations[3],
            norm_type=norm_type,
        )
        self.avg_conv = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            ConvBlock(
                channels,
                inner_channels,
                kernel_size=1,
                dilation=1,
                norm_type=norm_type,
                act_type=act_type,
            ),
        )
        self.transition = ConvBlock(
            cat_channels,
            channels,
            kernel_size=1,
            dilation=dilations[0],
            norm_type=norm_type,
            act_type=act_type,
        )

    def forward(self, input):
        aspp0 = self.aspp0(input)
        aspp1 = self.aspp1(input)
        aspp2 = self.aspp2(input)
        aspp3 = self.aspp3(input)
        avg = self.avg_conv(input)
        avg = F.interpolate(avg, aspp2.size()[2:], mode="nearest")
        out = torch.cat((aspp0, aspp1, aspp2, aspp3, avg), dim=1)
        out = self.transition(out)
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


class ClsRegHead(nn.Module):
    def __init__(
        self,
        in_channels,
        feature_size=96,
        conv_num=2,
        norm_type="GN",
        act_type="LeakyReLU",
    ):
        super(ClsRegHead, self).__init__()

        conv_s = []
        for i in range(conv_num):
            if i == 0:
                conv_s.append(
                    ConvBlock(
                        in_channels,
                        feature_size,
                        3,
                        norm_type=norm_type,
                        act_type=act_type,
                    )
                )
            else:
                conv_s.append(
                    ConvBlock(
                        feature_size,
                        feature_size,
                        3,
                        norm_type=norm_type,
                        act_type=act_type,
                    )
                )
        self.conv_s = nn.Sequential(*conv_s)
        self.cls_output = nn.Conv3d(feature_size, 1, kernel_size=3, padding=1)

        conv_r = []
        for i in range(conv_num):
            if i == 0:
                conv_r.append(
                    ConvBlock(
                        in_channels,
                        feature_size,
                        3,
                        norm_type=norm_type,
                        act_type=act_type,
                    )
                )
            else:
                conv_r.append(
                    ConvBlock(
                        feature_size,
                        feature_size,
                        3,
                        norm_type=norm_type,
                        act_type=act_type,
                    )
                )
        self.conv_r = nn.Sequential(*conv_r)
        self.shape_output = nn.Conv3d(feature_size, 3, kernel_size=3, padding=1)

        conv_o = []
        for i in range(conv_num):
            if i == 0:
                conv_o.append(
                    ConvBlock(
                        in_channels,
                        feature_size,
                        3,
                        norm_type=norm_type,
                        act_type=act_type,
                    )
                )
            else:
                conv_o.append(
                    ConvBlock(
                        feature_size,
                        feature_size,
                        3,
                        norm_type=norm_type,
                        act_type=act_type,
                    )
                )
        self.conv_o = nn.Sequential(*conv_o)
        self.offset_output = nn.Conv3d(feature_size, 3, kernel_size=3, padding=1)

    def forward(self, x):
        Shape = self.shape_output(self.conv_r(x))
        Offset = self.offset_output(self.conv_o(x))
        Cls = self.cls_output(self.conv_s(x))
        dict1 = {}
        dict1["Cls"] = Cls
        dict1["Shape"] = Shape
        dict1["Offset"] = Offset
        return dict1
