import torch.nn as nn
from .deform_conv_v3 import DeformConv2d

"""
    Here we have updated the implementation of 3D deformable convolutions to achieve 3D operations in a more lightweight manner.
"""
class DeformConv3d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, bias=False, modulation=True, channel_expand=3):
        super(DeformConv3d, self).__init__()
        self.inc = inc
        self.outc = outc
        self.channel_expand = channel_expand
        self.kernel_size = kernel_size
        self.padding = padding
        self.modulation = modulation

        self.spatial_conv = DeformConv2d(
            30, 30 * channel_expand, kernel_size,
            padding=padding, stride=1,
            bias=bias, modulation=modulation
        )

        self.channel_compress = nn.Conv2d(30 * channel_expand, 30, kernel_size=1, bias=bias)

        self.spectral_conv = nn.Sequential(
            nn.Conv3d(outc, outc,
                      kernel_size=(3,1,1),
                      padding=(1,0,0), stride=1, bias=False),
            nn.BatchNorm3d(outc),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * T, C, H, W)

        out = self.spatial_conv(x)
        out = self.channel_compress(out)

        out = out.view(B, T, C, H, W)
        out = self.spectral_conv(out)

        return out
