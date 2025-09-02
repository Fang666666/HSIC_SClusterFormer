import math
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from .FS_Attention import FreqSpectralAttentionLayer
from .deform_conv_v3 import DeformConv2d
from .Pseudo3DDeformConv import DeformConv3d
from .CrossAttention import FusionEncoder


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    """
    return pair-wise similarity matrix between two tensors
    :param x1: [B,...,M,D]
    :param x2: [B,...,N,D]
    :return: similarity matrix [B,...,M,N]
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim


def pairwise_euclidean_sim(x1: torch.Tensor, x2: torch.Tensor):
    """
    return pair-wise similarity matrix between two tensors
    :param x1: [B,...,M,D]
    :param x2: [B,...,N,D]
    :return: similarity matrix [B,...,M,N]
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)

    em = torch.norm(x1.unsqueeze(-2) - x2.unsqueeze(-3), dim=-1)

    sim = torch.exp(-em)

    return sim


class Cluster3D(nn.Module):
    def __init__(self, patch_size=13, dim=256, out_dim=256, proposal_w=2, proposal_h=2, fold_w=1, fold_h=1, heads=4,
                 head_dim=24, return_center=False):
        super().__init__()
        self.patch_size = patch_size
        self.heads = heads
        self.head_dim = head_dim
        self.f = nn.Conv2d(dim, heads * head_dim, kernel_size=1)
        self.proj = nn.Conv2d(heads * head_dim, out_dim, kernel_size=1)
        self.v = nn.Conv2d(dim, heads * head_dim, kernel_size=1)
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        self.centers_proposal = nn.AdaptiveAvgPool3d((1, proposal_w, proposal_h))
        self.fold_w = fold_w
        self.fold_h = fold_h
        self.return_center = return_center
        self.rule1 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4)
        )

    def forward(self, x):  # [b, n, c]
        x = rearrange(x, "b (w h) c -> b c w h", w=self.patch_size, h=self.patch_size)
        value = self.v(x)
        x = self.f(x)
        x = rearrange(x, "b (e c) w h -> (b e) c w h", e=self.heads)
        value = rearrange(value, "b (e c) w h -> (b e) c w h", e=self.heads)
        if self.fold_w > 1 and self.fold_h > 1:
            b0, c0, w0, h0 = x.shape
            assert w0 % self.fold_w == 0 and h0 % self.fold_h == 0, \
                f"Ensure the feature map size ({w0}*{h0}) can be divided by fold {self.fold_w}*{self.fold_h}"
            x = rearrange(x, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w,
                          f2=self.fold_h)  # [bs*blocks,c,ks[0],ks[1]]
            value = rearrange(value, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w, f2=self.fold_h)
        b, c, w, h = x.shape
        x = x.unsqueeze(2)
        value = value.unsqueeze(2)
        centers = rearrange(self.centers_proposal(x),
                            'b c d w h -> b (c d) w h')
        value_centers = rearrange(self.centers_proposal(value), 'b c d w h -> b (w h) (c d)')  # [b,C_W,C_H,c]
        b, c, ww, hh = centers.shape
        sim = self.rule1(
            self.sim_beta +
            self.sim_alpha * pairwise_cos_sim(
                centers.reshape(b, c, -1).permute(0, 2, 1),
                x.reshape(b, c, -1).permute(0, 2, 1)
            )
        )
        sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)
        mask = torch.zeros_like(sim)
        mask.scatter_(1, sim_max_idx, 1.)
        sim = sim * mask
        value2 = rearrange(value, 'b c d w h -> b (w h) (c d)')
        out = ((value2.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2) + value_centers) / (mask.sum(dim=-1, keepdim=True) + 1.0)

        if self.return_center:
            out = rearrange(out, "b (w h) c -> b c w h", w=ww)
        else:
            out = (out.unsqueeze(dim=2) * sim.unsqueeze(dim=-1)).sum(dim=1)
            out = rearrange(out, "b (w h) c -> b c w h", w=w)

        if self.fold_w > 1 and self.fold_h > 1:
            out = rearrange(out, "(b f1 f2) c w h -> b c (f1 w) (f2 h)", f1=self.fold_w, f2=self.fold_h)
        out = rearrange(out, "(b e) c w h -> b (e c) w h", e=self.heads)
        out = self.proj(out)
        out = rearrange(out, "b c w h -> b (w h) c")
        return out


class Cluster2D(nn.Module):
    def __init__(self, patch_size, dim=768, out_dim=768, proposal_w=2, proposal_h=2, fold_w=2, fold_h=2, heads=4,
                 head_dim=24,
                 return_center=False):
        super().__init__()
        self.patch_size = patch_size
        self.heads = heads
        self.head_dim = head_dim
        self.f = nn.Conv2d(dim, heads * head_dim, kernel_size=1)
        self.proj = nn.Conv2d(heads * head_dim, out_dim, kernel_size=1)
        self.v = nn.Conv2d(dim, heads * head_dim, kernel_size=1)
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        self.centers_proposal = nn.AdaptiveAvgPool2d((proposal_w, proposal_h))
        self.fold_w = fold_w
        self.fold_h = fold_h
        self.return_center = return_center
        self.rule2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4)
        )

    def forward(self, x):
        x = rearrange(x, "b (w h) c -> b c w h", w=self.patch_size, h=self.patch_size)
        value = self.v(x)
        x = self.f(x)
        x = rearrange(x, "b (e c) w h -> (b e) c w h", e=self.heads)
        value = rearrange(value, "b (e c) w h -> (b e) c w h", e=self.heads)
        if self.fold_w > 1 and self.fold_h > 1:
            b0, c0, w0, h0 = x.shape
            assert w0 % self.fold_w == 0 and h0 % self.fold_h == 0, \
                f"Ensure the feature map size ({w0}*{h0}) can be divided by fold {self.fold_w}*{self.fold_h}"
            x = rearrange(x, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w,
                          f2=self.fold_h)
            value = rearrange(value, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w, f2=self.fold_h)
        b, c, w, h = x.shape
        centers = self.centers_proposal(x)
        value_centers = rearrange(self.centers_proposal(value), 'b c w h -> b (w h) c')
        b, c, ww, hh = centers.shape
        sim = self.rule2(
            self.sim_beta +
            self.sim_alpha * pairwise_euclidean_sim(
                centers.reshape(b, c, -1).permute(0, 2, 1),
                x.reshape(b, c, -1).permute(0, 2, 1)
            )
        )
        sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)
        mask = torch.zeros_like(sim)
        mask.scatter_(1, sim_max_idx, 1.)
        sim = sim * mask
        value2 = rearrange(value, 'b c w h -> b (w h) c')
        out = ((value2.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2) + value_centers) / (
                mask.sum(dim=-1, keepdim=True) + 1.0)

        if self.return_center:
            out = rearrange(out, "b (w h) c -> b c w h", w=ww)
        else:
            out = (out.unsqueeze(dim=2) * sim.unsqueeze(dim=-1)).sum(dim=1)
            out = rearrange(out, "b (w h) c -> b c w h", w=w)

        if self.fold_w > 1 and self.fold_h > 1:
            out = rearrange(out, "(b f1 f2) c w h -> b c (f1 w) (f2 h)", f1=self.fold_w, f2=self.fold_h)
        out = rearrange(out, "(b e) c w h -> b (e c) w h", e=self.heads)
        out = self.proj(out)
        out = rearrange(out, "b c w h -> b (w h) c")
        return out


class GroupedPixelEmbedding(nn.Module):
    def __init__(self, in_feature_map_size=7, in_chans=3, embed_dim=128, n_groups=1):
        super().__init__()
        self.ifm_size = in_feature_map_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1, groups=n_groups)
        self.batch_norm = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.proj(x)
        x = self.relu(self.batch_norm(x))

        x = x.flatten(2).transpose(1, 2)

        after_feature_map_size = self.ifm_size

        return x, after_feature_map_size


class PixelEmbedding(nn.Module):
    def __init__(self, in_feature_map_size=7, in_chans=3, embed_dim=128, n_groups=1, i=0):
        super().__init__()
        self.ifm_size = in_feature_map_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1 if i == 0 else 2,
                              padding=1 if i == 0 else (3 // 2, 3 // 2))
        self.batch_norm = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.proj(x)
        x = self.relu(self.batch_norm(x))

        after_feature_map_size = x.shape[2]

        x = x.flatten(2).transpose(1, 2)

        return x, after_feature_map_size


class Block(nn.Module):
    def __init__(self, patch_size, dim, num_heads, mlp_ratio=4, drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Cluster3D(patch_size=patch_size, dim=dim, out_dim=dim, proposal_w=4, proposal_h=4, fold_w=1,
                              fold_h=1, heads=num_heads, head_dim=24)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Block2D(nn.Module):
    def __init__(self, patch_size, dim, num_heads, mlp_ratio=4, drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Cluster2D(patch_size=patch_size, dim=dim, out_dim=dim, proposal_w=4, proposal_h=4, fold_w=1,
                              fold_h=1, heads=num_heads, head_dim=24)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

"""
    Here we have updated the implementation of multi-scale operations, achieving refined feature extraction through a lighter-weight shared weight approach.
"""
class MultiScaleDeformConv3D_FSA(nn.Module):
    def __init__(self, deform_conv: nn.Module, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.deform_conv = deform_conv
        self.out_channels = deform_conv.outc * 30

        self.scale_convs = nn.ModuleList([
            nn.Conv3d(deform_conv.outc, deform_conv.outc, kernel_size=1, groups=deform_conv.outc)
            for _ in kernel_sizes
        ])

        self.fuse = nn.Conv3d(
            deform_conv.outc * len(kernel_sizes),
            deform_conv.outc,
            kernel_size=1
        )

        self.attention = FreqSpectralAttentionLayer(
            channel=self.out_channels,
            dct_h=kernel_sizes[-1],
            dct_w=kernel_sizes[-1],
            reduction=16,
            freq_sel_method='top2'
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        feat_base = self.deform_conv(x)
        feats = []
        for idx, k in enumerate(self.kernel_sizes):
            if k == self.kernel_sizes[0]:
                feat_scaled = feat_base
            else:
                scale = k / self.kernel_sizes[0]
                feat_scaled = F.avg_pool3d(
                    feat_base,
                    kernel_size=(1, int(scale), int(scale)),
                    stride=(1, int(scale), int(scale)),
                    ceil_mode=True
                )
                feat_scaled = F.interpolate(
                    feat_scaled, size=(D, H, W),
                    mode='trilinear', align_corners=False
                )

            feat_scaled = self.scale_convs[idx](feat_scaled)
            feats.append(feat_scaled)

        multi_scale = torch.cat(feats, dim=1)
        multi_scale = self.fuse(multi_scale)

        out = feat_base + multi_scale

        B, Ck, D, H, W = out.shape
        out_4d = out.view(B, Ck * D, H, W)
        out_attn = self.attention(out_4d)
        out_final = out_attn.view(B, Ck, D, H, W)

        return out_final



class MultiScaleDeformConv2D(nn.Module):
    def __init__(self, deform_conv: nn.Module, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.deform_conv = deform_conv
        self.fuse = nn.Conv2d(deform_conv.outc * 3, deform_conv.outc, kernel_size=1)

    def forward(self, x):  # x: [B, C, H, W]
        B, C, H, W = x.shape
        feats = []

        for k in self.kernel_sizes:
            scale = k / self.kernel_sizes[0]
            if scale != 1.0:
                x_scaled = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
            else:
                x_scaled = x

            feat = self.deform_conv(x_scaled)
            feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
            feats.append(feat)

        out = torch.cat(feats, dim=1)
        out = self.fuse(out) + feats[1]

        return out


class SClusterFormer(nn.Module):
    def __init__(self, img_size=224, pca_components=3, emap_components=1, num_classes=1000, num_stages=3,
                 n_groups=[32, 32, 32], embed_dims=[256, 128, 64], num_heads=[8, 8, 8], mlp_ratios=[1, 1, 1],
                 depths=[2, 2, 2], patchsize=17):
        super().__init__()
        self.reducedbands = pca_components
        self.num_stages = num_stages

        new_bands = math.ceil(pca_components / n_groups[0]) * n_groups[0]
        self.pad = nn.ReplicationPad3d((0, 0, 0, 0, 0, new_bands - pca_components))

        """MDC-FSA"""
        deform_conv_shared = DeformConv3d(inc=1, outc=1, kernel_size=3, padding=1, bias=False, modulation=True)
        self.deform_conv_layer_pca = MultiScaleDeformConv3D_FSA(deform_conv_shared)

        deform_conv_shared_emap = DeformConv2d(inc=emap_components, outc=30, kernel_size=9, padding=1, bias=False, modulation=True)
        self.deform_conv_layer_emap = MultiScaleDeformConv2D(deform_conv_shared_emap)

        """Upper Branch"""
        for i in range(num_stages):
            patch_embed = GroupedPixelEmbedding(
                in_feature_map_size=img_size,
                in_chans=new_bands if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],
                n_groups=n_groups[i]
            )

            block = nn.ModuleList([Block(
                dim=embed_dims[i],
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratios[i],
                drop=0.,
                patch_size=img_size) for j in range(depths[i])])

            norm2d = nn.LayerNorm(embed_dims[i])

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm2d)

        """Lower Branch"""
        self.embed_img = [img_size, math.ceil(img_size / 2), math.ceil(math.ceil(img_size / 2) / 2)]
        for i in range(num_stages):
            patch_embed2d = PixelEmbedding(
                in_feature_map_size=img_size if i == 0 else self.embed_img[i - 1],
                in_chans=new_bands if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],
                n_groups=n_groups[i],
                i=i
            )

            block2d = nn.ModuleList([Block2D(
                dim=embed_dims[i],
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratios[i],
                drop=0.,
                patch_size=self.embed_img[i]) for j in range(depths[i])])

            norm = nn.LayerNorm(embed_dims[i])

            setattr(self, f"patch_embed2d{i + 1}", patch_embed2d)
            setattr(self, f"block2d{i + 1}", block2d)
            setattr(self, f"norm2d{i + 1}", norm)

        """CFAF"""
        self.coefficients = torch.nn.Parameter(torch.Tensor([0.7]))

        self.fusion_encoder = FusionEncoder(
            depth=1,
            h_dim=64,
            ct_attn_heads=4,
            ct_attn_depth=1,
            dropout=0.1,
            patchsize=patchsize
        )

        self.head = nn.Sequential(
            nn.Linear(embed_dims[-1], num_classes),
            nn.Softmax(dim=1)
        )

    def forward_features_Upper(self, x):
        x = self.pad(x).squeeze(dim=1)
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")

            x, s = patch_embed(x)
            for blk in block:
                x = blk(x)

            x = norm(x)

            if i != self.num_stages - 1:
                x = x.reshape(B, s, s, -1).permute(0, 3, 1, 2).contiguous()

        return x

    def forward_features_Lower(self, x):
        x = self.pad(x).squeeze(dim=1)
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed2d{i + 1}")
            block = getattr(self, f"block2d{i + 1}")
            norm = getattr(self, f"norm2d{i + 1}")

            x, s = patch_embed(x)
            for blk in block:
                x = blk(x)

            x = norm(x)

            if i != self.num_stages - 1:
                x = x.reshape(B, s, s, -1).permute(0, 3, 1, 2).contiguous()

        return x


    def forward(self, x):
        x1_pca = x[:, :, :self.reducedbands, :, :]
        x1_pca = self.deform_conv_layer_pca(x1_pca)
        x1_pca = x1_pca[:, :self.reducedbands, :, :]

        x_scluster = self.forward_features_Upper(x1_pca)

        x_emap = x[:, :, self.reducedbands, :, :]
        x_emap = self.deform_conv_layer_emap(x_emap)
        x_emap = torch.unsqueeze(x_emap, dim=1)

        x_emap = self.forward_features_Lower(x_emap)

        x_cfpf = self.fusion_encoder(x_scluster, x_emap)

        x_cfpf = x_cfpf.mean(dim=1)
        x_scluster = x_scluster.mean(dim=1)
        x_emap = x_emap.mean(dim=1)

        x_scluster = self.head(x_scluster)
        x_emap = self.head(x_emap)
        x_cfpf = self.head(x_cfpf)

        x = x_emap * ((1 - self.coefficients) / 2) + x_cfpf * (
                (1 - self.coefficients) / 2) + x_scluster * self.coefficients

        return x

