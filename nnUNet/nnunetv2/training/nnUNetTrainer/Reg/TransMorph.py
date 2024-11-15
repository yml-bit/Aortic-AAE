'''
TransMorph model
Modified and tested by:
'''

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_, to_3tuple
from torch.distributions.normal import Normal
import torch.nn.functional as nnf
import numpy as np
import configs_TransMorph as configs
from networks import Unet, ConvBlock
import pdb
import sys
import os
sys.path.insert(0, os.path.abspath('.'))
import sys
sys.path.append(r"/media/bit301/data/yml/project/python39/p2/nnUNet/nnunetv2/training/nnUNetTrainer/Reg")
import layers
import torch.nn.functional as F
from mamba import MambaLayer
try:
    from mamba import *
except ModuleNotFoundError:
    pass


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, L, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, window_size, C)
    """
    B, H, W, L, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], L // window_size[2],
               window_size[2], C)

    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows


def window_reverse(windows, window_size, H, W, L):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
        L (int): Length of image
    Returns:
        x: (B, H, W, L, C)
    """
    B = int(windows.shape[0] / (H * W * L / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(B, H // window_size[0], W // window_size[1], L // window_size[2], window_size[0], window_size[1],
                     window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, H, W, L, -1)
    return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, rpe=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1 * 2*Wt-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_t = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_t]))  # 3, Wh, Ww, Wt
        coords_flatten = torch.flatten(coords, 1)  # 3, Wh*Ww*Wt
        self.rpe = rpe
        if self.rpe:
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wh*Ww*Wt, Wh*Ww*Wt
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww*Wt, Wh*Ww*Wt, 3
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww*Wt, Wh*Ww*Wt
            self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww, Wt*Ww) or None
        """
        B_, N, C = x.shape  # (num_windows*B, Wh*Ww*Wt, C)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] * self.window_size[2],
                self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  # Wh*Ww*Wt,Wh*Ww*Wt,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww*Wt, Wh*Ww*Wt
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=(7, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, rpe=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= min(self.shift_size) < min(
            self.window_size), "shift_size must in 0-window_size, shift_sz: {}, win_size: {}".format(self.shift_size,
                                                                                                     self.window_size)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, rpe=rpe, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None
        self.T = None

    def forward(self, x, mask_matrix):
        H, W, T = self.H, self.W, self.T
        B, L, C = x.shape
        assert L == H * W * T, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, T, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_f = 0
        pad_r = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_b = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_h = (self.window_size[2] - T % self.window_size[2]) % self.window_size[2]
        x = nnf.pad(x, (0, 0, pad_f, pad_h, pad_t, pad_b, pad_l, pad_r))
        _, Hp, Wp, Tp, _ = x.shape

        # cyclic shift
        if min(self.shift_size) > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                                   dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2],
                                   C)  # nW*B, window_size*window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp, Tp)  # B H' W' L' C

        # reverse cyclic shift
        if min(self.shift_size) > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                           dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0 or pad_h > 0:
            x = x[:, :H, :W, :T, :].contiguous()

        x = x.view(B, H * W * T, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, reduce_factor=2):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, (8 // reduce_factor) * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x, H, W, T):
        """
        x: B, H*W*T, C
        """
        B, L, C = x.shape
        assert L == H * W * T, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0 and T % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, T, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1) or (T % 2 == 1)
        if pad_input:
            x = nnf.pad(x, (0, 0, 0, T % 2, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, 0::2, :]  # B H/2 W/2 T/2 C
        x1 = x[:, 1::2, 0::2, 0::2, :]  # B H/2 W/2 T/2 C
        x2 = x[:, 0::2, 1::2, 0::2, :]  # B H/2 W/2 T/2 C
        x3 = x[:, 0::2, 0::2, 1::2, :]  # B H/2 W/2 T/2 C
        x4 = x[:, 1::2, 1::2, 0::2, :]  # B H/2 W/2 T/2 C
        x5 = x[:, 0::2, 1::2, 1::2, :]  # B H/2 W/2 T/2 C
        x6 = x[:, 1::2, 0::2, 1::2, :]  # B H/2 W/2 T/2 C
        x7 = x[:, 1::2, 1::2, 1::2, :]  # B H/2 W/2 T/2 C
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B H/2 W/2 T/2 8*C
        x = x.view(B, -1, 8 * C)  # B H/2*W/2*T/2 8*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 rpe=True,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 pat_merg_rf=2, ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.pat_merg_rf = pat_merg_rf
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else (
                    window_size[0] // 2, window_size[1] // 2, window_size[2] // 2),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                rpe=rpe,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, reduce_factor=self.pat_merg_rf)
        else:
            self.downsample = None

    def forward(self, x, H, W, T):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size[0])) * self.window_size[0]
        Wp = int(np.ceil(W / self.window_size[1])) * self.window_size[1]
        Tp = int(np.ceil(T / self.window_size[2])) * self.window_size[2]
        img_mask = torch.zeros((1, Hp, Wp, Tp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))
        w_slices = (slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None))
        t_slices = (slice(0, -self.window_size[2]),
                    slice(-self.window_size[2], -self.shift_size[2]),
                    slice(-self.shift_size[2], None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                for t in t_slices:
                    img_mask[:, h, w, t, :] = cnt
                    cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W, blk.T = H, W, T
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W, T)
            Wh, Ww, Wt = (H + 1) // 2, (W + 1) // 2, (T + 1) // 2
            return x, H, W, T, x_down, Wh, Ww, Wt
        else:
            return x, H, W, T, x, H, W, T


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W, T = x.size()
        if T % self.patch_size[2] != 0:
            x = nnf.pad(x, (0, self.patch_size[2] - T % self.patch_size[2]))
        if W % self.patch_size[1] != 0:
            x = nnf.pad(x, (0, 0, 0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = nnf.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        x = self.proj(x)  # B C Wh Ww Wt
        if self.norm is not None:
            Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww, Wt)

        return x


class SinusoidalPositionEmbedding(nn.Module):
    '''
    Rotary Position Embedding
    '''

    def __init__(self, ):
        super(SinusoidalPositionEmbedding, self).__init__()

    def forward(self, x):
        batch_sz, n_patches, hidden = x.shape
        position_ids = torch.arange(0, n_patches).float().cuda()
        indices = torch.arange(0, hidden // 2).float().cuda()
        indices = torch.pow(10000.0, -2 * indices / hidden)
        embeddings = torch.einsum('b,d->bd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, (1, n_patches, hidden))
        return embeddings


class SinPositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(SinPositionalEncoding3D, self).__init__()
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        self.inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        # self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        tensor = tensor.permute(0, 2, 3, 4, 1)
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)
        emb = torch.zeros((x, y, z, self.channels * 3), device=tensor.device).type(tensor.type())
        emb[:, :, :, :self.channels] = emb_x
        emb[:, :, :, self.channels:2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels:] = emb_z
        emb = emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)
        return emb.permute(0, 4, 1, 2, 3)


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (tuple): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 spe=False,
                 rpe=True,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 pat_merg_rf=2, ):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.spe = spe
        self.rpe = rpe
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_3tuple(self.pretrain_img_size)
            patch_size = to_3tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1],
                                  pretrain_img_size[2] // patch_size[2]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        elif self.spe:
            self.pos_embd = SinPositionalEncoding3D(embed_dim).cuda()
            # self.pos_embd = SinusoidalPositionEmbedding().cuda()
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias,
                               rpe=rpe,
                               qk_scale=qk_scale,
                               drop=drop_rate,
                               attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               pat_merg_rf=pat_merg_rf, )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)

        Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)

        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = nnf.interpolate(self.absolute_pos_embed, size=(Wh, Ww, Wt), mode='trilinear')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        elif self.spe:
            x = (x + self.pos_embd(x)).flatten(2).transpose(1, 2)
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, T, x, Wh, Ww, Wt = layer(x, Wh, Ww, Wt)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, T, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()
                outs.append(out)
        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()


class MambaBlock(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        drop_rate (float): Dropout rate. Default: 0
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 drop_rate=0.,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 spe=False,
                 rpe=True,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 ):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.spe = spe
        self.rpe = rpe
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_3tuple(self.pretrain_img_size)
            patch_size = to_3tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1],
                                  pretrain_img_size[2] // patch_size[2]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        elif self.spe:
            self.pos_embd = SinPositionalEncoding3D(embed_dim).cuda()
        self.pos_drop = nn.Dropout(p=drop_rate)

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = MambaLayer(dim=int(embed_dim * 2 ** i_layer),
                               d_state=d_state,
                               d_conv=d_conv,
                               expand=expand,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)

        Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)

        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = nnf.interpolate(self.absolute_pos_embed, size=(Wh, Ww, Wt), mode='trilinear')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        elif self.spe:
            x = (x + self.pos_embd(x)).flatten(2).transpose(1, 2)
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, T, x, Wh, Ww, Wt = layer(x, Wh, Ww, Wt)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, T, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()
                outs.append(out)
        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(MambaBlock, self).train(mode)
        self._freeze_stages()


class VMambaBlock(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        drop_rate (float): Dropout rate. Default: 0
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 2],
                 drop_rate=0.,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 spe=False,
                 rpe=True,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 ):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.depths = depths
        self.embed_dim = embed_dim
        self.ape = ape
        self.spe = spe
        self.rpe = rpe
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_3tuple(self.pretrain_img_size)
            patch_size = to_3tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1],
                                  pretrain_img_size[2] // patch_size[2]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        elif self.spe:
            self.pos_embd = SinPositionalEncoding3D(embed_dim).cuda()
        self.pos_drop = nn.Dropout(p=drop_rate)

        # build layers
        for i_layer in range(self.num_layers):
            self.layers = nn.ModuleList()
            for i_layer in range(self.num_layers):
                layer = VMambaLayer(dim=int(embed_dim * 2 ** i_layer),
                                depths = self.depths[i_layer],
                                d_state=d_state,
                                d_conv=d_conv,
                                expand=expand,
                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None)
                self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x) # torch.Size([1, 96, 12, 12, 12])
        Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)

        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = nnf.interpolate(self.absolute_pos_embed, size=(Wh, Ww, Wt), mode='trilinear')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        elif self.spe:
            x = (x + self.pos_embd(x)).flatten(2).transpose(1, 2)
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)
        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, T, x, Wh, Ww, Wt = layer(x, Wh, Ww, Wt)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, T, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()
                outs.append(out)
        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(VMambaBlock, self).train(mode)
        self._freeze_stages()



class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        relu = nn.LeakyReLU(inplace=True)
        if not use_batchnorm:
            nm = nn.InstanceNorm3d(out_channels)
        else:
            nm = nn.BatchNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, nm, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class RegistrationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
        conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        super().__init__(conv3d)


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    Obtained from https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

#other voxemorph
# class UNet(nn.Module):
#     def contracting_block(self, in_channels, out_channels, kernel_size=3):
#         """
#         This function creates one contracting block
#         """
#         block = torch.nn.Sequential(
#             torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
#             torch.nn.BatchNorm3d(out_channels),
#             torch.nn.ReLU(),
#             torch.nn.Conv3d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1),
#             torch.nn.BatchNorm3d(out_channels),
#             torch.nn.ReLU(),
#         )
#         return block
#
#     def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
#         """
#         This function creates one expansive block
#         """
#         block = torch.nn.Sequential(
#             torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
#             torch.nn.BatchNorm3d(mid_channel),
#             torch.nn.ReLU(),
#             torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
#             torch.nn.BatchNorm3d(mid_channel),
#             torch.nn.ReLU(),
#             torch.nn.ConvTranspose3d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
#             torch.nn.BatchNorm3d(out_channels),
#             torch.nn.ReLU(),
#         )
#         return block
#
#     def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
#         """
#         This returns final block
#         """
#         block = torch.nn.Sequential(
#                     torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
#                     torch.nn.BatchNorm3d(mid_channel),
#                     torch.nn.ReLU(),
#                     torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
#                     torch.nn.BatchNorm3d(out_channels),
#                     torch.nn.ReLU()
#                 )
#         return block
#
#     def __init__(self, in_channel, out_channel):
#         super(UNet, self).__init__()
#         #Encode
#         self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=32)
#         self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
#         self.conv_encode2 = self.contracting_block(32, 64)
#         self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
#         self.conv_encode3 = self.contracting_block(64, 128)
#         self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
#         # Bottleneck
#         mid_channel = 128
#         self.bottleneck = torch.nn.Sequential(
#                                 torch.nn.Conv3d(kernel_size=3, in_channels=mid_channel, out_channels=mid_channel * 2, padding=1),
#                                 torch.nn.BatchNorm3d(mid_channel * 2),
#                                 torch.nn.ReLU(),
#                                 torch.nn.Conv3d(kernel_size=3, in_channels=mid_channel*2, out_channels=mid_channel, padding=1),
#                                 torch.nn.BatchNorm3d(mid_channel),
#                                 torch.nn.ReLU(),
#                                 torch.nn.ConvTranspose3d(in_channels=mid_channel, out_channels=mid_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
#                                 torch.nn.BatchNorm3d(mid_channel),
#                                 torch.nn.ReLU(),
#                             )
#         # Decode
#         self.conv_decode3 = self.expansive_block(256, 128, 64)
#         self.conv_decode2 = self.expansive_block(128, 64, 32)
#         self.final_layer = self.final_block(64, 32, out_channel)
#
#     def crop_and_concat(self, upsampled, bypass, crop=False):
#         """
#         This layer crop the layer from contraction block and concat it with expansive block vector
#         """
#         if crop:
#             c = (bypass.size()[2] - upsampled.size()[2]) // 2
#             bypass = F.pad(bypass, (-c, -c, -c, -c))
#         return torch.cat((upsampled, bypass), 1)
#
#     def forward(self, x):
#         # Encode
#         encode_block1 = self.conv_encode1(x)
#         encode_pool1 = self.conv_maxpool1(encode_block1)
#         encode_block2 = self.conv_encode2(encode_pool1)
#         encode_pool2 = self.conv_maxpool2(encode_block2)
#         encode_block3 = self.conv_encode3(encode_pool2)
#         encode_pool3 = self.conv_maxpool3(encode_block3)
#         # Bottleneck
#         bottleneck1 = self.bottleneck(encode_pool3)
#         # Decode
#         decode_block3 = self.crop_and_concat(bottleneck1, encode_block3)
#         cat_layer2 = self.conv_decode3(decode_block3)
#         decode_block2 = self.crop_and_concat(cat_layer2, encode_block2)
#         cat_layer1 = self.conv_decode2(decode_block2)
#         decode_block1 = self.crop_and_concat(cat_layer1, encode_block1)
#         final_layer = self.final_layer(decode_block1)
#         return  final_layer
#
# class VoxelMorph3d(nn.Module):
#     def __init__(self, config):
#         super(VoxelMorph3d, self).__init__()
#         self.unet = UNet(1, 3)
#         self.reg_head = RegistrationHead(
#             in_channels=config.reg_head_chan,
#             out_channels=3,
#             kernel_size=3,
#         )
#         self.spatial_trans = SpatialTransformer(config.img_size)
#     def forward(self, source, target):
#         x = torch.cat([source, target], dim=3).permute(0,3,1,2)
#         x = self.unet(x).permute(0,2,3,1)
#         flow = self.reg_head(x)
#         moved = self.spatial_trans(source, flow)
#         ret = {'moved_vol': moved, 'preint_flow': flow}
#         return ret
#
# class VoxelMorph3d_feat(nn.Module):
#     def __init__(self, config):
#         super(VoxelMorph3d_feat, self).__init__()
#         nb_feat_extractor = [[16] * 2, [16] * 4]
#         self.feature_extractor = Unet(config.img_size,
#                                       infeats=1,
#                                       nb_features=nb_feat_extractor,
#                                       nb_levels=None,
#                                       feat_mult=1,
#                                       nb_conv_per_level=1,
#                                       half_res=False)
#         self.unet = UNet(1, 3)
#         self.reg_head = RegistrationHead(
#             in_channels=config.reg_head_chan,
#             out_channels=3,
#             kernel_size=3,
#         )
#         self.spatial_trans = SpatialTransformer(config.img_size)
#
#     def forward(self, source, fixed_image):
#         source_feat = self.feature_extractor(source)
#         target_feat = self.feature_extractor(fixed_image)
#
#         x = torch.cat([source_feat, target_feat], dim=1)
#         # x = torch.cat([moving_image, fixed_image], dim=3).permute(0,3,1,2)
#         x = self.unet(x).permute(0,2,3,1)
#         flow = self.reg_head(x)
#         moved = self.spatial_trans(source, flow)
#         ret = {'moved_vol': moved, 'preint_flow': flow}
#         return ret

################
class TransMorph(nn.Module):
    def __init__(self, config):
        '''
        TransMorph Model
        '''
        super(TransMorph, self).__init__()
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer(patch_size=config.patch_size,
                                           in_chans=config.in_chans,
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           rpe=config.rpe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           )

        self.up1 = DecoderBlock(embed_dim * 4, embed_dim * 2, skip_channels=embed_dim * 2 if if_transskip else 0,
                                use_batchnorm=False)  # 384, 20, 20, 64
        self.up2 = DecoderBlock(embed_dim * 2, embed_dim, skip_channels=embed_dim if if_transskip else 0,
                                use_batchnorm=False)  # 384, 40, 40, 64
        self.up3 = DecoderBlock(embed_dim, embed_dim // 2, skip_channels=embed_dim // 2 if if_convskip else 0,
                                use_batchnorm=False)  # 384, 80, 80, 128
        self.up4 = DecoderBlock(embed_dim // 2, config.reg_head_chan,
                                skip_channels=config.reg_head_chan if if_convskip else 0,
                                use_batchnorm=False)  # 384, 160, 160, 256
        self.c1 = Conv3dReLU(2, embed_dim // 2, 3, 1, use_batchnorm=False)
        self.c2 = Conv3dReLU(2, config.reg_head_chan, 3, 1, use_batchnorm=False)
        self.reg_head = RegistrationHead(
            in_channels=config.reg_head_chan,
            out_channels=3,
            kernel_size=3,
        )
        self.spatial_trans = SpatialTransformer(config.img_size)
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)

    def forward(self, source, target, return_pos_flow=False, return_feature=False):

        x = torch.cat([source, target], dim=1)

        if self.if_convskip:
            x_s0 = x.clone()
            x_s1 = self.avg_pool(x)
            f4 = self.c1(x_s1)
            f5 = self.c2(x_s0)
        else:
            f4 = None
            f5 = None

        out_feats = self.transformer(x)

        if self.if_transskip:
            f1 = out_feats[-2]
            f2 = out_feats[-3]
        else:
            f1 = None
            f2 = None

        x = self.up1(out_feats[-1], f1)
        x = self.up2(x, f2)
        # x = self.up2(x, f3)
        x = self.up3(x, f4)
        x = self.up4(x, f5)
        flow = self.reg_head(x)
        moved = self.spatial_trans(source, flow)
        ret = {'moved_vol': moved, 'preint_flow': flow}

        return ret

class MambaMorph(nn.Module):
    def __init__(self, config):
        """
        MambaMorph Model
        """
        super(MambaMorph, self).__init__()
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = MambaBlock(patch_size=config.patch_size,
                                      in_chans=config.in_chans,
                                      embed_dim=config.embed_dim,
                                      depths=config.depths,
                                      drop_rate=config.drop_rate,
                                      ape=config.ape,
                                      spe=config.spe,
                                      rpe=config.rpe,
                                      patch_norm=config.patch_norm,
                                      out_indices=config.out_indices,
                                      d_state=config.d_state,
                                      d_conv=config.d_conv,
                                      expand=config.expand,
                                      )

        self.up1 = DecoderBlock(embed_dim * 4, embed_dim * 2, skip_channels=embed_dim * 2 if if_transskip else 0,
                                use_batchnorm=False)  # 384, 20, 20, 64
        self.up2 = DecoderBlock(embed_dim * 2, embed_dim, skip_channels=embed_dim if if_transskip else 0,
                                use_batchnorm=False)  # 384, 40, 40, 64
        self.up3 = DecoderBlock(embed_dim, embed_dim // 2, skip_channels=embed_dim // 2 if if_convskip else 0,
                                use_batchnorm=False)  # 384, 80, 80, 128
        self.up4 = DecoderBlock(embed_dim // 2, config.reg_head_chan,
                                skip_channels=config.reg_head_chan if if_convskip else 0,
                                use_batchnorm=False)  # 384, 160, 160, 256
        self.c1 = Conv3dReLU(2, embed_dim // 2, 3, 1, use_batchnorm=False)
        self.c2 = Conv3dReLU(2, config.reg_head_chan, 3, 1, use_batchnorm=False)
        self.reg_head = RegistrationHead(
            in_channels=config.reg_head_chan,
            out_channels=3,
            kernel_size=3,
        )
        self.spatial_trans = SpatialTransformer(config.img_size)
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        self.integrate = layers.VecInt(config.img_size, 7)

    # def forward(self, source, target, return_feature=False):
    def forward(self, source, target, return_pos_flow=True, return_feature=False):

        x = torch.cat([source, target], dim=1)

        if self.if_convskip:
            x_s0 = x.clone()
            x_s1 = self.avg_pool(x)
            f4 = self.c1(x_s1)
            f5 = self.c2(x_s0)
        else:
            f4 = None
            f5 = None

        out_feats = self.transformer(x)

        if self.if_transskip:
            f1 = out_feats[-2]
            f2 = out_feats[-3]
        else:
            f1 = None
            f2 = None

        x = self.up1(out_feats[-1], f1)
        x = self.up2(x, f2)

        x = self.up3(x, f4)
        x = self.up4(x, f5)
        flow = self.reg_head(x)
        pos_flow = self.integrate(flow)  #
        # moved = self.spatial_trans(source, flow)
        moved = self.spatial_trans(source, pos_flow)  #
        
        ret = {'moved_vol': moved, 'preint_flow': flow}
        if return_pos_flow:
            ret['pos_flow'] = pos_flow
        return ret

class MambaMorphOri(nn.Module):
    def __init__(self, config):
        """
        MambaMorph Model
        """
        super(MambaMorphOri, self).__init__()
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = MambaBlock(patch_size=config.patch_size,
                                      in_chans=config.in_chans,
                                      embed_dim=config.embed_dim,
                                      depths=config.depths,
                                      drop_rate=config.drop_rate,
                                      ape=config.ape,
                                      spe=config.spe,
                                      rpe=config.rpe,
                                      patch_norm=config.patch_norm,
                                      out_indices=config.out_indices,
                                      d_state=config.d_state,
                                      d_conv=config.d_conv,
                                      expand=config.expand,
                                      )

        self.up1 = DecoderBlock(embed_dim * 4, embed_dim * 2, skip_channels=embed_dim * 2 if if_transskip else 0,
                                use_batchnorm=False)  # 384, 20, 20, 64
        self.up2 = DecoderBlock(embed_dim * 2, embed_dim, skip_channels=embed_dim if if_transskip else 0,
                                use_batchnorm=False)  # 384, 40, 40, 64
        self.up3 = DecoderBlock(embed_dim, embed_dim // 2, skip_channels=embed_dim // 2 if if_convskip else 0,
                                use_batchnorm=False)  # 384, 80, 80, 128
        self.up4 = DecoderBlock(embed_dim // 2, config.reg_head_chan,
                                skip_channels=config.reg_head_chan if if_convskip else 0,
                                use_batchnorm=False)  # 384, 160, 160, 256
        self.c1 = Conv3dReLU(2, embed_dim // 2, 3, 1, use_batchnorm=False)
        self.c2 = Conv3dReLU(2, config.reg_head_chan, 3, 1, use_batchnorm=False)
        self.reg_head = RegistrationHead(
            in_channels=config.reg_head_chan,
            out_channels=3,
            kernel_size=3,
        )
        self.spatial_trans = SpatialTransformer(config.img_size)
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)

    def forward(self, source, target, return_pos_flow=True, return_feature=False):

        x = torch.cat([source, target], dim=1)

        if self.if_convskip:
            x_s0 = x.clone()
            x_s1 = self.avg_pool(x)
            f4 = self.c1(x_s1)
            f5 = self.c2(x_s0)
        else:
            f4 = None
            f5 = None

        out_feats = self.transformer(x)

        if self.if_transskip:
            f1 = out_feats[-2]
            f2 = out_feats[-3]
        else:
            f1 = None
            f2 = None

        x = self.up1(out_feats[-1], f1)
        x = self.up2(x, f2)

        x = self.up3(x, f4)
        x = self.up4(x, f5)
        flow = self.reg_head(x)
        moved = self.spatial_trans(source, flow)

        ret = {'moved_vol': moved, 'preint_flow': flow}

        return ret

class TransMorphFeat(nn.Module):
    def __init__(self, config):
        '''
        TransMorph Model
        '''
        super(TransMorphFeat, self).__init__()
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        nb_feat_extractor = [[16] * 2, [16] * 4]
        self.feature_extractor = Unet(config.img_size,
                                      infeats=1,
                                      nb_features=nb_feat_extractor,
                                      nb_levels=None,
                                      feat_mult=1,
                                      nb_conv_per_level=1,
                                      half_res=False)
        self.transformer = SwinTransformer(patch_size=config.patch_size,
                                           in_chans=nb_feat_extractor[-1][-1] * 2,
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           rpe=config.rpe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           )

        self.up1 = DecoderBlock(embed_dim * 4, embed_dim * 2, skip_channels=embed_dim * 2 if if_transskip else 0,
                                use_batchnorm=False)  # 384, 20, 20, 64
        self.up2 = DecoderBlock(embed_dim * 2, embed_dim, skip_channels=embed_dim if if_transskip else 0,
                                use_batchnorm=False)  # 384, 40, 40, 64
        self.up3 = DecoderBlock(embed_dim, embed_dim // 2, skip_channels=embed_dim // 2 if if_convskip else 0,
                                use_batchnorm=False)  # 384, 80, 80, 128
        self.up4 = DecoderBlock(embed_dim // 2, config.reg_head_chan,
                                skip_channels=config.reg_head_chan if if_convskip else 0,
                                use_batchnorm=False)  # 384, 160, 160, 256
        self.c1 = Conv3dReLU(nb_feat_extractor[-1][-1] * 2, embed_dim // 2, 3, 1, use_batchnorm=False)
        self.c2 = Conv3dReLU(nb_feat_extractor[-1][-1] * 2, config.reg_head_chan, 3, 1, use_batchnorm=False)
        self.reg_head = RegistrationHead(
            in_channels=config.reg_head_chan,
            out_channels=3,
            kernel_size=3,
        )
        self.spatial_trans = SpatialTransformer(config.img_size)
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)

    def forward(self, source, target, return_pos_flow=False, return_feature=False):
        source_feat = self.feature_extractor(source)
        target_feat = self.feature_extractor(target)

        x = torch.cat([source_feat, target_feat], dim=1)

        if self.if_convskip:
            x_s0 = x.clone()
            x_s1 = self.avg_pool(x)
            f4 = self.c1(x_s1)
            f5 = self.c2(x_s0)
        else:
            f4 = None
            f5 = None

        out_feats = self.transformer(x)

        if self.if_transskip:
            f1 = out_feats[-2]
            f2 = out_feats[-3]
        else:
            f1 = None
            f2 = None

        x = self.up1(out_feats[-1], f1)
        x = self.up2(x, f2)
        # x = self.up2(x, f3)
        x = self.up3(x, f4)
        x = self.up4(x, f5)
        flow = self.reg_head(x)
        moved = self.spatial_trans(source, flow)
        ret = {'moved_vol': moved, 'preint_flow': flow}
        return ret

class MambaMorphFeat(nn.Module):
    def __init__(self, config):
        """
        MambaMorph Model
        """
        super(MambaMorphFeat, self).__init__()
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        nb_feat_extractor = [[16] * 2, [16] * 4]
        self.feature_extractor = Unet(config.img_size,
                                      infeats=1,
                                      nb_features=nb_feat_extractor,
                                      nb_levels=None,
                                      feat_mult=1,
                                      nb_conv_per_level=1,
                                      half_res=False)

        # ReadOut head for feature constrastive learning
        self.read_out_head = nn.ModuleList()

        self.read_out_head.append(ConvBlock(len(config.img_size), nb_feat_extractor[-1][-1],
                                            nb_feat_extractor[-1][-1] // 2, stride=2))
        self.read_out_head.append(ConvBlock(len(config.img_size), nb_feat_extractor[-1][-1] // 2, 3))

        self.transformer = MambaBlock(patch_size=config.patch_size,
                                      in_chans=nb_feat_extractor[-1][-1] * 2,
                                      embed_dim=config.embed_dim,
                                      depths=config.depths,
                                      drop_rate=config.drop_rate,
                                      ape=config.ape,
                                      spe=config.spe,
                                      rpe=config.rpe,
                                      patch_norm=config.patch_norm,
                                      out_indices=config.out_indices,
                                      d_state=config.d_state,
                                      d_conv=config.d_conv,
                                      expand=config.expand,
                                      )

        self.up1 = DecoderBlock(embed_dim * 4, embed_dim * 2, skip_channels=embed_dim * 2 if if_transskip else 0,
                                use_batchnorm=False)  # 384, 20, 20, 64
        self.up2 = DecoderBlock(embed_dim * 2, embed_dim, skip_channels=embed_dim if if_transskip else 0,
                                use_batchnorm=False)  # 384, 40, 40, 64
        self.up3 = DecoderBlock(embed_dim, embed_dim // 2, skip_channels=embed_dim // 2 if if_convskip else 0,
                                use_batchnorm=False)  # 384, 80, 80, 128
        self.up4 = DecoderBlock(embed_dim // 2, config.reg_head_chan,
                                skip_channels=config.reg_head_chan if if_convskip else 0,
                                use_batchnorm=False)  # 384, 160, 160, 256
        self.c1 = Conv3dReLU(nb_feat_extractor[-1][-1] * 2, embed_dim // 2, 3, 1, use_batchnorm=False)
        self.c2 = Conv3dReLU(nb_feat_extractor[-1][-1] * 2, config.reg_head_chan, 3, 1, use_batchnorm=False)
        self.reg_head = RegistrationHead(
            in_channels=config.reg_head_chan,
            out_channels=3,
            kernel_size=3,
        )
        self.spatial_trans = SpatialTransformer(config.img_size)
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        self.integrate = layers.VecInt(config.img_size, 7)

    def forward(self, source, target, return_pos_flow=True, return_feature=False):
        source_feat = self.feature_extractor(source)
        target_feat = self.feature_extractor(target)

        x = torch.cat([source_feat, target_feat], dim=1)

        if self.if_convskip:
            x_s0 = x.clone()
            x_s1 = self.avg_pool(x)
            f4 = self.c1(x_s1)
            f5 = self.c2(x_s0)
        else:
            f4 = None
            f5 = None

        out_feats = self.transformer(x)

        if self.if_transskip:
            f1 = out_feats[-2]
            f2 = out_feats[-3]
        else:
            f1 = None
            f2 = None

        x = self.up1(out_feats[-1], f1)
        x = self.up2(x, f2)

        x = self.up3(x, f4)
        x = self.up4(x, f5)
        flow = self.reg_head(x)
        pos_flow = self.integrate(flow)
        moved = self.spatial_trans(source, pos_flow)

        ret = {'moved_vol': moved, 'preint_flow': flow}
        if return_pos_flow:
            ret['pos_flow'] = pos_flow
        if return_feature:
            for head in self.read_out_head:
                source_feat = head(source_feat)
                target_feat = head(target_feat)
            ret['feature'] = torch.cat((source_feat, target_feat), dim=1)
        return ret

class VMambaMorph(nn.Module):
    def __init__(self, config):
        """
        VMambaMorph Model
        """
        super(VMambaMorph, self).__init__()
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = VMambaBlock(patch_size=config.patch_size,
                                      in_chans=config.in_chans,
                                      embed_dim=config.embed_dim,
                                      depths=config.depths,
                                      drop_rate=config.drop_rate,
                                      ape=config.ape,
                                      spe=config.spe,
                                      rpe=config.rpe,
                                      patch_norm=config.patch_norm,
                                      out_indices=config.out_indices,
                                      d_state=config.d_state,
                                      d_conv=config.d_conv,
                                      expand=config.expand,
                                      )

        self.up1 = DecoderBlock(embed_dim * 4, embed_dim * 2, skip_channels=embed_dim * 2 if if_transskip else 0,
                                use_batchnorm=False)  # 384, 20, 20, 64
        self.up2 = DecoderBlock(embed_dim * 2, embed_dim, skip_channels=embed_dim if if_transskip else 0,
                                use_batchnorm=False)  # 384, 40, 40, 64
        self.up3 = DecoderBlock(embed_dim, embed_dim // 2, skip_channels=embed_dim // 2 if if_convskip else 0,
                                use_batchnorm=False)  # 384, 80, 80, 128
        self.up4 = DecoderBlock(embed_dim // 2, config.reg_head_chan,
                                skip_channels=config.reg_head_chan if if_convskip else 0,
                                use_batchnorm=False)  # 384, 160, 160, 256
        self.c1 = Conv3dReLU(2, embed_dim // 2, 3, 1, use_batchnorm=False)
        self.c2 = Conv3dReLU(2, config.reg_head_chan, 3, 1, use_batchnorm=False)
        self.reg_head = RegistrationHead(
            in_channels=config.reg_head_chan,
            out_channels=3,
            kernel_size=3,
        )
        self.spatial_trans = SpatialTransformer(config.img_size)
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        self.integrate = layers.VecInt(config.img_size, 7)

    def forward(self, source, target, return_pos_flow=True, return_feature=False):

        x = torch.cat([source, target], dim=1)

        if self.if_convskip:
            x_s0 = x.clone()
            x_s1 = self.avg_pool(x)
            f4 = self.c1(x_s1)
            f5 = self.c2(x_s0)
        else:
            f4 = None
            f5 = None

        out_feats = self.transformer(x)

        if self.if_transskip:
            f1 = out_feats[-2]
            f2 = out_feats[-3]
        else:
            f1 = None
            f2 = None

        x = self.up1(out_feats[-1], f1)
        x = self.up2(x, f2)

        x = self.up3(x, f4)
        x = self.up4(x, f5)
        flow = self.reg_head(x)
        pos_flow = self.integrate(flow)  #
        # moved = self.spatial_trans(source, flow)
        moved = self.spatial_trans(source, pos_flow)  #
        
        ret = {'moved_vol': moved, 'preint_flow': flow}
        if return_pos_flow:
            ret['pos_flow'] = pos_flow
        return ret


class VMambaMorphFeat(nn.Module):
    def __init__(self, config):
        """
        VMambaMorph Model
        """
        super(VMambaMorphFeat, self).__init__()
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        nb_feat_extractor = [[16] * 2, [16] * 4]
        self.feature_extractor = Unet(config.img_size,
                                      infeats=1,
                                      nb_features=nb_feat_extractor,
                                      nb_levels=None,
                                      feat_mult=1,
                                      nb_conv_per_level=1,
                                      half_res=False)

        # ReadOut head for feature constrastive learning
        self.read_out_head = nn.ModuleList()

        self.read_out_head.append(ConvBlock(len(config.img_size), nb_feat_extractor[-1][-1],
                                            nb_feat_extractor[-1][-1] // 2, stride=2))
        self.read_out_head.append(ConvBlock(len(config.img_size), nb_feat_extractor[-1][-1] // 2, 3))

        self.transformer = VMambaBlock(patch_size=config.patch_size,
                                      in_chans=nb_feat_extractor[-1][-1] * 2,
                                      embed_dim=config.embed_dim,
                                      depths=config.depths,
                                      drop_rate=config.drop_rate,
                                      ape=config.ape,
                                      spe=config.spe,
                                      rpe=config.rpe,
                                      patch_norm=config.patch_norm,
                                      out_indices=config.out_indices,
                                      d_state=config.d_state,
                                      d_conv=config.d_conv,
                                      expand=config.expand,
                                      )

        self.up1 = DecoderBlock(embed_dim * 4, embed_dim * 2, skip_channels=embed_dim * 2 if if_transskip else 0,
                                use_batchnorm=False)  # 384, 20, 20, 64
        self.up2 = DecoderBlock(embed_dim * 2, embed_dim, skip_channels=embed_dim if if_transskip else 0,
                                use_batchnorm=False)  # 384, 40, 40, 64
        self.up3 = DecoderBlock(embed_dim, embed_dim // 2, skip_channels=embed_dim // 2 if if_convskip else 0,
                                use_batchnorm=False)  # 384, 80, 80, 128
        
        self.up4 = DecoderBlock(embed_dim // 2, config.reg_head_chan,
                                skip_channels=config.reg_head_chan if if_convskip else 0,
                                use_batchnorm=False)  # 384, 160, 160, 256
        self.c1 = Conv3dReLU(nb_feat_extractor[-1][-1] * 2, embed_dim // 2, 3, 1, use_batchnorm=False)
        self.c2 = Conv3dReLU(nb_feat_extractor[-1][-1] * 2, config.reg_head_chan, 3, 1, use_batchnorm=False)
        self.reg_head = RegistrationHead(
            in_channels=config.reg_head_chan,
            out_channels=3,
            kernel_size=3,
        )
        self.spatial_trans = SpatialTransformer(config.img_size)
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        self.integrate = layers.VecInt(config.img_size, 7)


    def forward(self, source, target, return_pos_flow=True, return_feature=False):
        source_feat = self.feature_extractor(source)
        target_feat = self.feature_extractor(target)

        x = torch.cat([source_feat, target_feat], dim=1)
        if self.if_convskip:
            x_s0 = x.clone()
            x_s1 = self.avg_pool(x)
            f4 = self.c1(x_s1)
            f5 = self.c2(x_s0)
        else:
            f4 = None
            f5 = None

        out_feats = self.transformer(x) 

        if self.if_transskip:
            f1 = out_feats[-2] 
            f2 = out_feats[-3] 
 
        else:
            f1 = None
            f2 = None

        x = self.up1(out_feats[-1], f1)
        x = self.up2(x, f2)
        # print(x.shape)
        x = self.up3(x, f4)
        # print(x.shape)
        x = self.up4(x, f5)
        # print(x.shape)
        flow = self.reg_head(x) 
        pos_flow = self.integrate(flow)
        # print(pos_flow.shape)
        moved = self.spatial_trans(source, pos_flow)

        ret = {'moved_vol': moved, 'preint_flow': flow}
        if return_pos_flow:
            ret['pos_flow'] = pos_flow
        if return_feature:
            for head in self.read_out_head:
                source_feat = head(source_feat)
                target_feat = head(target_feat)
            ret['feature'] = torch.cat((source_feat, target_feat), dim=1)
        return ret

'''
class RecVMambaMorphFeat(nn.Module):
    def __init__(self, config):
        """
        RecVMambaMorph Model
        """
        super(RecVMambaMorphFeat, self).__init__()
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        nb_feat_extractor = [[16] * 2, [16] * 4]
        self.feature_extractor = Unet(config.img_size,
                                      infeats=1,
                                      nb_features=nb_feat_extractor,
                                      nb_levels=None,
                                      feat_mult=1,
                                      nb_conv_per_level=1,
                                      half_res=False)

        # ReadOut head for feature constrastive learning
        self.read_out_head = nn.ModuleList()

        self.read_out_head.append(ConvBlock(len(config.img_size), nb_feat_extractor[-1][-1],
                                            nb_feat_extractor[-1][-1] // 2, stride=2))
        self.read_out_head.append(ConvBlock(len(config.img_size), nb_feat_extractor[-1][-1] // 2, 3))

        self.transformer = VMambaBlock(patch_size=config.patch_size,
                                       in_chans=nb_feat_extractor[-1][-1] * 2,
                                       embed_dim=config.embed_dim,
                                       depths=config.depths,
                                       drop_rate=config.drop_rate,
                                       ape=config.ape,
                                       spe=config.spe,
                                       rpe=config.rpe,
                                       patch_norm=config.patch_norm,
                                       out_indices=config.out_indices,
                                       d_state=config.d_state,
                                       d_conv=config.d_conv,
                                       expand=config.expand,
                                       )

        self.up1 = DecoderBlock(embed_dim * 4, embed_dim * 2, skip_channels=embed_dim * 2 if if_transskip else 0,
                                use_batchnorm=False)  # 384, 20, 20, 64
        self.up2 = DecoderBlock(embed_dim * 2, embed_dim, skip_channels=embed_dim if if_transskip else 0,
                                use_batchnorm=False)  # 384, 40, 40, 64
        self.up3 = DecoderBlock(embed_dim, embed_dim // 2, skip_channels=embed_dim // 2 if if_convskip else 0,
                                use_batchnorm=False)  # 384, 80, 80, 128

        self.up4 = DecoderBlock(embed_dim // 2, config.reg_head_chan,
                                skip_channels=config.reg_head_chan if if_convskip else 0,
                                use_batchnorm=False)  # 384, 160, 160, 256
        self.c1 = Conv3dReLU(nb_feat_extractor[-1][-1] * 2, embed_dim // 2, 3, 1, use_batchnorm=False)
        self.c2 = Conv3dReLU(nb_feat_extractor[-1][-1] * 2, config.reg_head_chan, 3, 1, use_batchnorm=False)
        self.reg_head = RegistrationHead(
            in_channels=config.reg_head_chan,
            out_channels=3,
            kernel_size=3,
        )
        self.spatial_trans = SpatialTransformer(config.img_size)
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        self.integrate = layers.VecInt(config.img_size, 7)

    def forward(self, source, target, return_pos_flow=True, return_feature=False, rec_num=2):
        moved=source
        for rr in range(rec_num):
            source_feat = self.feature_extractor(moved)
            target_feat = self.feature_extractor(target)
            if rr==0:
                src_feat = source_feat
            x = torch.cat([source_feat, target_feat], dim=1)
            if self.if_convskip:
                x_s0 = x.clone()
                x_s1 = self.avg_pool(x)
                f4 = self.c1(x_s1)
                f5 = self.c2(x_s0)
            else:
                f4 = None
                f5 = None

            out_feats = self.transformer(x)

            if self.if_transskip:
                f1 = out_feats[-2]
                f2 = out_feats[-3]

            else:
                f1 = None
                f2 = None

            x = self.up1(out_feats[-1], f1)
            x = self.up2(x, f2)
            # print(x.shape)
            x = self.up3(x, f4)
            # print(x.shape)
            x = self.up4(x, f5)
            # print(x.shape)
            flow = self.reg_head(x)
            pos_flow = self.integrate(flow)
            # print(pos_flow.shape)
            disp_comp = pos_flow if rr==0 else self.spatial_trans(disp_comp, pos_flow) + pos_flow
            moved = self.spatial_trans(source, disp_comp)

        ret = {'moved_vol': moved, 'preint_flow': flow}
        if return_pos_flow:
            ret['pos_flow'] = disp_comp
        if return_feature:
            for head in self.read_out_head:
                source_feat = head(src_feat)
                target_feat = head(target_feat)
            ret['feature'] = torch.cat((source_feat, target_feat), dim=1)
        return ret
'''
class RecVMambaMorphFeat(nn.Module):
    def __init__(self, config):
        """
        RecVMambaMorph Model
        """
        super(RecVMambaMorphFeat, self).__init__()
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        nb_feat_extractor = [[16] * 2, [16] * 4]
        self.feature_extractor = Unet(config.img_size,
                                      infeats=1,
                                      nb_features=nb_feat_extractor,
                                      nb_levels=None,
                                      feat_mult=1,
                                      nb_conv_per_level=1,
                                      half_res=False)

        # ReadOut head for feature constrastive learning
        self.read_out_head = nn.ModuleList()

        self.read_out_head.append(ConvBlock(len(config.img_size), nb_feat_extractor[-1][-1],
                                            nb_feat_extractor[-1][-1] // 2, stride=2))
        self.read_out_head.append(ConvBlock(len(config.img_size), nb_feat_extractor[-1][-1] // 2, 3))

        self.transformer = VMambaBlock(patch_size=config.patch_size,
                                       in_chans=nb_feat_extractor[-1][-1] * 2,
                                       embed_dim=config.embed_dim,
                                       depths=config.depths,
                                       drop_rate=config.drop_rate,
                                       ape=config.ape,
                                       spe=config.spe,
                                       rpe=config.rpe,
                                       patch_norm=config.patch_norm,
                                       out_indices=config.out_indices,
                                       d_state=config.d_state,
                                       d_conv=config.d_conv,
                                       expand=config.expand,
                                       )

        self.up1 = DecoderBlock(embed_dim * 4, embed_dim * 2, skip_channels=embed_dim * 2 if if_transskip else 0,
                                use_batchnorm=False)  # 384, 20, 20, 64
        self.up2 = DecoderBlock(embed_dim * 2, embed_dim, skip_channels=embed_dim if if_transskip else 0,
                                use_batchnorm=False)  # 384, 40, 40, 64
        self.up3 = DecoderBlock(embed_dim, embed_dim // 2, skip_channels=embed_dim // 2 if if_convskip else 0,
                                use_batchnorm=False)  # 384, 80, 80, 128

        self.up4 = DecoderBlock(embed_dim // 2, config.reg_head_chan,
                                skip_channels=config.reg_head_chan if if_convskip else 0,
                                use_batchnorm=False)  # 384, 160, 160, 256
        self.c1 = Conv3dReLU(nb_feat_extractor[-1][-1] * 2, embed_dim // 2, 3, 1, use_batchnorm=False)
        self.c2 = Conv3dReLU(nb_feat_extractor[-1][-1] * 2, config.reg_head_chan, 3, 1, use_batchnorm=False)
        self.reg_head = RegistrationHead(
            in_channels=config.reg_head_chan,
            out_channels=3,
            kernel_size=3,
        )
        self.spatial_trans = SpatialTransformer(config.img_size)
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        self.integrate = layers.VecInt(config.img_size, 7)

    def feat_loss(self,feat1,feat2):

        return torch.mean(torch.abs(feat1 - feat2 - torch.mean(feat1,dim=[2,3,4],keepdim=True) + torch.mean(feat2,dim=[2,3,4],keepdim=True)))

    def forward(self, source, target, return_pos_flow=True, return_feature=False,return_feature_loss=True, rec_num=2):
        moved=source
        for rr in range(rec_num):
            source_feat = self.feature_extractor(moved)
            target_feat = self.feature_extractor(target)
            if rr==0:
                src_feat = source_feat
            # if rr == rec_num-1:
            #     loss_feat = self.feat_loss(self.avg_pool(source_feat),self.avg_pool(target_feat))
            x = torch.cat([source_feat, target_feat], dim=1)
            if self.if_convskip:
                x_s0 = x.clone()
                x_s1 = self.avg_pool(x)
                f4 = self.c1(x_s1)
                f5 = self.c2(x_s0)
            else:
                f4 = None
                f5 = None

            out_feats = self.transformer(x)

            if self.if_transskip:
                f1 = out_feats[-2]
                f2 = out_feats[-3]

            else:
                f1 = None
                f2 = None

            x = self.up1(out_feats[-1], f1)
            x = self.up2(x, f2)
            # print(x.shape)
            x = self.up3(x, f4)
            # print(x.shape)
            x = self.up4(x, f5)
            # print(x.shape)
            flow = self.reg_head(x)
            pos_flow = self.integrate(flow)
            # print(pos_flow.shape)
            disp_comp = pos_flow if rr==0 else self.spatial_trans(disp_comp, pos_flow) + pos_flow
            moved = self.spatial_trans(source, disp_comp)

        ret = {'moved_vol': moved, 'preint_flow': flow,'pos_flow':disp_comp}
        # if return_pos_flow:
        #     ret['pos_flow'] = disp_comp
        if return_feature_loss:
            ret['feature_loss'] = self.feat_loss(self.avg_pool(source_feat),self.avg_pool(target_feat))
        if return_feature:
            for head in self.read_out_head:
                source_feat = head(src_feat)
                target_feat = head(target_feat)
            ret['feature'] = torch.cat((source_feat, target_feat), dim=1)
        return ret

        
CONFIGS = {
    'TransMorph': configs.get_3DTransMorph_config(),
    'TransMorph-No-Conv-Skip': configs.get_3DTransMorphNoConvSkip_config(),
    'TransMorph-No-Trans-Skip': configs.get_3DTransMorphNoTransSkip_config(),
    'TransMorph-No-Skip': configs.get_3DTransMorphNoSkip_config(),
    'TransMorph-Lrn': configs.get_3DTransMorphLrn_config(),
    'TransMorph-Sin': configs.get_3DTransMorphSin_config(),
    'TransMorph-No-RelPosEmbed': configs.get_3DTransMorphNoRelativePosEmbd_config(),
    'TransMorph-Large': configs.get_3DTransMorphLarge_config(),
    'TransMorph-Small': configs.get_3DTransMorphSmall_config(),
    'TransMorph-Tiny': configs.get_3DTransMorphTiny_config(),
    'MambaMorph': configs.get_3DMambaMorph_config(),
    'VMambaMorph': configs.get_3DVMambaMorph_config(),
    'RecVMambaMorph': configs.get_3DVMambaMorph_config(),
}


# from TransMorph import CONFIGS as CONFIGS_TM
# if __name__ =='__main__':
#     config = CONFIGS_TM['MambaMorph']
#     model = MambaMorphFeat(config=config).to('cuda')
    # print("11")
    # source = torch.randn(1,1,48,48,48).cuda()
    # target = torch.randn(1,1,48,48,48).cuda()
    # output = model(source, target,True,True)
    # print(output['moved_vol'].shape)
    # print(output['preint_flow'].shape)
    # print(output['feature'].shape)
