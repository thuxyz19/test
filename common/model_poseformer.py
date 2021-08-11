## Our PoseFormer model was revised from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

import math
import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import numpy as np
import time
import copy


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


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., factor=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class RelativeAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., seq=9):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.B = nn.Parameter(torch.randn(1, num_heads, seq, seq), requires_grad=True)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale + self.B

        if mask is not None:
            mask = mask.view(B, 1, N, N)
            attn = mask + attn

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class PureAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qk_scale=None, attn_drop=0., seq=9):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.B = nn.Parameter(torch.randn(1, num_heads, seq, seq), requires_grad=True)

    def forward(self, q, k, v, mask=None):
        # q, k, v: (B, H, F, C)
        attn = (q @ k.transpose(-2, -1)) * self.scale + self.B

        if mask is not None:
            mask = mask.view(attn.shape[0], 1, attn.shape[2], attn.shape[3])
            attn = mask + attn

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v

        return x

class GroupAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., seq=9):
        super().__init__()
        self.groups = ((1, 2, 3), (4, 5, 6), (0, 7, 8, 9, 10), (11, 12, 13), (14, 15, 16))
        if len(self.groups) > 0:
            self.inverse_group = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            count = 0
            for i in range(len(self.groups)):
                for g_idx in self.groups[i]:
                    self.inverse_group[g_idx] = count
                    count += 1
            self.inverse_group = torch.tensor(self.inverse_group).type(torch.long).cuda(non_blocking=True)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim*17, dim * 17 * 3, bias=qkv_bias)
        self.attn = nn.ModuleList([
            PureAttention(dim * len(self.groups[i]), num_heads=num_heads, qk_scale=qk_scale, attn_drop=attn_drop, seq=seq)
            for i in range(len(self.groups))
        ])
        self.proj = nn.Linear(dim*17, dim*17)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, 17, self.num_heads, -1).permute(2, 0, 4, 1, 3, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        B, H, N = q.shape[:3]
        x = []
        for i in range(len(self.groups)):
            q_i = q[:, :, :, self.groups[i], :].view(*q.shape[:3], -1)
            k_i = k[:, :, :, self.groups[i], :].view(*k.shape[:3], -1)
            v_i = v[:, :, :, self.groups[i], :].view(*v.shape[:3], -1)
            x_i = self.attn[i](q_i, k_i, v_i, mask)
            x.append(x_i.view(*x_i.shape[:3], len(self.groups[i]), -1))
        x = torch.cat(x, 3)
        x = x[:, :, :, self.inverse_group, :].permute(0, 2, 3, 1, 4).view(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class RelativeDualAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., seq=9, F=999):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.B = nn.Parameter(torch.randn(1, num_heads, seq, seq + F), requires_grad=True)

    def forward(self, x, ref_2D):
        ## x: (b, f, c), ref_2D: (1, F, c)
        B, N, C = x.shape
        _, F, _ = ref_2D.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        ref_qkv = self.qkv(ref_2D).reshape(1, F, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        ref_k, ref_v = ref_qkv[1], ref_qkv[2]
        ref_k = ref_k.repeat(B, 1, 1, 1)
        ref_v = ref_v.repeat(B, 1, 1, 1)
        k = torch.cat([k, ref_k], 2)
        v = torch.cat([v, ref_v], 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale + self.B

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class RelativeDualShortAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., seq=9):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.B = nn.Parameter(torch.randn(1, num_heads, seq, 2*seq), requires_grad=True)

    def forward(self, x, ref_2D):
        ## x: (b, f, c), ref_2D: (b, f, c)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        ref_qkv = self.qkv(ref_2D).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        ref_q, ref_k, ref_v = ref_qkv[0], ref_qkv[1], ref_qkv[2]

        k_tmp = torch.cat([k, ref_k], 2)
        v_tmp = torch.cat([v, ref_v], 2)

        attn = (q @ k_tmp.transpose(-2, -1)) * self.scale + self.B

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v_tmp).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        k_tmp = torch.cat([ref_k, k], 2)
        v_tmp = torch.cat([ref_v, v], 2)

        attn = (ref_q @ k_tmp.transpose(-2, -1)) * self.scale + self.B

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        ref_2D = (attn @ v_tmp).transpose(1, 2).reshape(B, N, C)

        ref_2D = self.proj(ref_2D)
        ref_2D = self.proj_drop(ref_2D)


        return x, ref_2D

class ProbSparseAttention(nn.Module):
    def __init__(self, dim, factor_q=1, factor_k=5, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # parameters for ProbSparse self-attention
        self.factor_q = int(factor_q)
        self.factor_k = int(factor_k)

    def forward(self, x):
        B, N, C = x.shape
        U = np.round(np.log(N) / np.log(3)).astype('int').item()
        u_top_q = self.factor_q * U
        u_top_k = self.factor_k * U
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # sample the k matrix
        index = torch.randint(0, N, (u_top_q,), device='cuda')
        # q_sample = torch.gather(q, dim=2, index=index.view(1,1,u_top_q,1).repeat(q.shape[0], q.shape[1], 1, q.shape[-1]))
        q_sample = q[:, :, index, :]

        # compute the sample score
        start = time.time()
        score = q_sample @ k.transpose(-2, -1)
        end = time.time()
        interval1 = end - start
        # measurement = torch.max(score, dim=-1)[0] - torch.mean(score, dim=-1)
        measurement = torch.sum(score, dim=-2)  # (B, H, N)

        # set u-top queries under M
        M_top = measurement.topk(u_top_k, dim=-1, sorted=False)[1]
        # q_sample = q[torch.arange(B)[:, None, None], torch.arange(self.num_heads)[None, :, None],
        #              M_top, :]
        k_sample = torch.gather(k, dim=2, index=M_top.unsqueeze(-1).repeat(1, 1, 1, k.shape[-1]))
        v_sample = torch.gather(v, dim=2, index=M_top.unsqueeze(-1).repeat(1, 1, 1, v.shape[-1]))

        start = time.time()
        # compute probSparse attention
        attn = (q @ k_sample.transpose(-2, -1)) * self.scale
        end = time.time()
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        interval2 = end - start

        start = time.time()
        # build new feature maps
        S = attn @ v_sample
        S = S.transpose(1, 2)
        end = time.time()
        # S0 = v.mean(dim=2, keepdim=True)
        # S = torch.cat((S1, S0), dim=2).transpose(1, 2)
        B_n, N_n, H_n, F_n = S.shape
        C_n = H_n * F_n
        x = S.reshape(B_n, N_n, C_n)
        interval3 = end - start

        x = self.proj(x)
        x = self.proj_drop(x)
        print(interval1, interval2, interval3)

        return x


class DownSampleAttention(nn.Module):
    def __init__(self, dim, down_factor=3, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # parameters for ProbSparse self-attention
        self.down_factor = int(down_factor)

    def forward(self, x):
        B, N, C = x.shape

        U = 5 * np.round(np.log(N) / np.log(3)).astype('int').item()

        x_top_k = N // self.down_factor

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # sample the k matrix
        index = torch.randint(0, N, (U,), device='cuda')
        # q_sample = torch.gather(q, dim=2, index=index.view(1,1,u_top_q,1).repeat(q.shape[0], q.shape[1], 1, q.shape[-1]))
        q_sample = q[:, :, index, :]

        # compute the sample score
        score = q_sample @ k.transpose(-2, -1)
        # measurement = torch.max(score, dim=-1)[0] - torch.mean(score, dim=-1)
        measurement = torch.sum(score, dim=-2)  # (B, H, N)
        measurement = measurement.mean(1)

        # set u-top queries under M
        M_top = measurement.topk(x_top_k, dim=-1, sorted=False)[1]
        # q_sample = q[torch.arange(B)[:, None, None], torch.arange(self.num_heads)[None, :, None],
        #              M_top, :]
        k_sample = torch.gather(k, dim=2, index=M_top.unsqueeze(1).unsqueeze(-1).repeat(1, k.shape[1], 1, k.shape[-1]))
        v_sample = torch.gather(v, dim=2, index=M_top.unsqueeze(1).unsqueeze(-1).repeat(1, v.shape[1], 1, v.shape[-1]))
        q_sample = torch.gather(q, dim=2, index=M_top.unsqueeze(1).unsqueeze(-1).repeat(1, q.shape[1], 1, q.shape[-1]))

        # compute probSparse attention
        attn = (q_sample @ k_sample.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # build new feature maps
        S = attn @ v_sample
        S = S.transpose(1, 2)
        # S0 = v.mean(dim=2, keepdim=True)
        # S = torch.cat((S1, S0), dim=2).transpose(1, 2)
        B_n, N_n, H_n, F_n = S.shape
        C_n = H_n * F_n
        x = S.reshape(B_n, N_n, C_n)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x, M_top


class TBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, down_factor=3):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = DownSampleAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            down_factor=down_factor)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        res, m_top = self.attn(self.norm1(x))
        x_sample = torch.gather(x, dim=1, index=m_top.unsqueeze(-1).repeat(1, 1, x.shape[-1]))
        x = x_sample + self.drop_path(res)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_type=Attention, factor_q=1, factor_k=3):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if attn_type is Attention:
            self.attn = attn_type(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        else:
            self.attn = attn_type(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, seq=9)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DualBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, f=9, F=999):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = RelativeDualAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, seq=f, F=F)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, ref_2D, x):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm2(ref_2D)))
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return ref_2D, x


class DualShortBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, f=9):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = RelativeDualShortAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, seq=f)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, ref_2D, x):
        x_tmp, ref_2D_tmp = self.attn(self.norm1(x), self.norm2(ref_2D))
        x = x + self.drop_path(x_tmp)
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        ref_2D = ref_2D + self.drop_path(ref_2D_tmp)
        ref_2D = ref_2D + self.drop_path(self.mlp(self.norm4(ref_2D)))
        return ref_2D, x



class ShiftParallelBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, M=9, shift=3, downsample=False):
        super().__init__()
        self.block = ShiftBlock(dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                 drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer, M=M, shift=shift, downsample=False)
        self.downsample = downsample

    def forward(self, ref_2D, x):
        # x: (b, f, c), ref_2D: (b, f, c)
        b, f, c = x.shape
        x_tmp = torch.cat([x, ref_2D], 0)
        x_tmp = self.block(x_tmp)
        x = x_tmp[:b, :, :].contiguous()
        ref_2D = x_tmp[b:, :, :].contiguous()
        if self.downsample:
            x = x.view(b, -1, 3, c).mean(2)
            ref_2D = ref_2D.view(b, -1, 3, c).mean(2)
        return ref_2D, x

class ParallelBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.block = Block(dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                 drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer, attn_type=RelativeAttention)

    def forward(self, ref_2D, x):
        # x: (b, f, c), ref_2D: (b, f, c)
        b, f, c = x.shape
        x_tmp = torch.cat([x, ref_2D], 0)
        x_tmp = self.block(x_tmp)
        x = x_tmp[:b, :, :].contiguous()
        ref_2D = x_tmp[b:, :, :].contiguous()
        return ref_2D, x

class RelativeBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, f=9):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        self.linear = nn.Linear(2*dim, dim)
        self.attn = RelativeAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, seq=f)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.head = nn.Linear(dim, 17*3)

    def forward(self, ref_2D, x):
        ref_2D_relative = self.linear(torch.cat([ref_2D, x]))

        ref_2D_tmp = self.attn(self.norm1(ref_2D_relative))
        ref_2D_relative = ref_2D_relative + self.drop_path(ref_2D_tmp)
        ref_2D_relative = ref_2D_relative + self.drop_path(self.mlp(self.norm2(ref_2D_relative)))

        x_relative = self.linear(torch.cat([x, ref_2D]))

        x_tmp = self.attn(self.norm3(x_relative))
        x_relative = x_relative + self.drop_path(x_tmp)
        x_relative = x_relative + self.drop_path(self.mlp(self.norm4(x_relative)))
        ref_2D_relative = ref_2D_relative.mean(1)
        x_relative = x_relative.mean(1)
        ref_relative = self.head(ref_2D_relative).view(-1, 1, 17, 3)
        x_relative = self.head(x_relative).view(-1, 1, 17, 3)


        return ref_relative, x_relative


class SwinBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, M=9, dilated_M=9, downsample=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = RelativeAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, seq=M)
        self.dilated_attn = RelativeAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, seq=dilated_M)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        self.dilated_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.M = M
        self.dilated_M = dilated_M
        self.downsample = downsample


    def forward(self, x):
        # x: (b, f, c)
        b, f, c = x.shape
        x = x.view(b, -1, self.M, c).view(-1, self.M, c)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.view(b, -1, self.M, c).view(b, f, c)
        x = x.view(b, self.dilated_M, -1, c).permute(0, 2, 1, 3).contiguous()
        x = x.view(-1, self.dilated_M, c)
        x = x + self.drop_path(self.dilated_attn(self.norm3(x)))
        x = x + self.drop_path(self.dilated_mlp(self.norm4(x)))
        x = x.view(b, -1, self.dilated_M, c).permute(0, 2, 1, 3).contiguous()
        x = x.view(b, f, c)
        if self.downsample:
            x = x.view(b, -1, 3, c).mean(2)
        return x


class ShiftBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, M=9, shift=3, downsample=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = RelativeAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, seq=M)
        self.norm2 = norm_layer(dim)
        self.shift_attn1 = RelativeAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, seq=M)
        self.norm3 = norm_layer(dim)
        self.shift_attn2 = RelativeAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, seq=M)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm4 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm5 = norm_layer(dim)
        self.shift_mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm6 = norm_layer(dim)
        self.shift_mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.M = M
        self.shift = shift
        self.downsample = downsample


    def shift_features(self, x):
        # x: (b, f, c)
        x = torch.cat([x[:, self.shift:, :], x[:, :self.shift, :]], 1)
        mask = torch.zeros(self.M, self.M, device=x.device, dtype=x.dtype)
        mask[:self.M - self.shift, self.M - self.shift:] = -np.inf
        mask[self.M - self.shift:, :self.M - self.shift] = -np.inf
        group_num = x.shape[1] // self.M
        mask_zeros = torch.zeros(group_num - 1, self.M, self.M, device=x.device, dtype=x.dtype)
        mask = torch.cat([mask_zeros, mask.view(1, self.M, self.M)], 0)
        return x, mask


    def forward(self, x):
        # x: (b, f, c)
        b, f, c = x.shape

        x = x.view(b, -1, self.M, c).view(-1, self.M, c)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm4(x)))
        x = x.view(b, -1, self.M, c).view(b, f, c)

        if f > 9:
            x, mask = self.shift_features(x)
        else:
            mask = torch.zeros(1, self.M, self.M, device=x.device, dtype=x.dtype)

        x = x.view(b, -1, self.M, c).view(-1, self.M, c)
        mask = mask.repeat(b, 1, 1)
        x = x + self.drop_path(self.shift_attn1(self.norm2(x), mask))
        x = x + self.drop_path(self.shift_mlp1(self.norm5(x)))
        x = x.view(b, -1, self.M, c).view(b, f, c)

        if f > 9:
            x, mask = self.shift_features(x)
        else:
            mask = torch.zeros(1, self.M, self.M, device=x.device, dtype=x.dtype)

        x = x.view(b, -1, self.M, c).view(-1, self.M, c)
        mask = mask.repeat(b, 1, 1)
        x = x + self.drop_path(self.shift_attn2(self.norm3(x), mask))
        x = x + self.drop_path(self.shift_mlp2(self.norm6(x)))
        x = x.view(b, -1, self.M, c).view(b, f, c)

        if f > 9:
            x = torch.cat([x[:, -self.shift*2:, :], x[:, :-self.shift*2, :]], 1)


        if self.downsample:
            x = x.view(b, -1, 3, c).mean(2)
        return x


class ShiftDualBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, M=9, shift=3, downsample=False):
        super().__init__()
        self.block = ShiftBlock(dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                 drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer, M=M, shift=shift, downsample=False)
        self.downsample = downsample

    def forward(self, ref_2D, x):
        # x: (b, f, c), ref_2D: (1, F, c)
        b, f, c = x.shape
        x = self.block(x)
        ref_2D = self.block(ref_2D)
        if self.downsample:
            x = x.view(b, -1, 3, c).mean(2)
        return ref_2D, x

class ShiftDualShortBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, M=9, shift=3, downsample=False):
        super().__init__()
        self.block = ShiftBlock(dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                 drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer, M=M, shift=shift, downsample=False)
        self.downsample = downsample

    def forward(self, ref_2D, x):
        # x: (b, f, c), ref_2D: (b, f, c)
        b, f, c = x.shape
        x_tmp = torch.cat([x, ref_2D], 0)
        x_tmp = self.block(x_tmp)
        x = x_tmp[:b, :, :].contiguous()
        ref_2D = x_tmp[b:, :, :].contiguous()
        if self.downsample:
            x = x.view(b, -1, 3, c).mean(2)
            ref_2D = ref_2D.view(b, -1, 3, c).mean(2)
        return ref_2D, x


class ShiftMutualBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, M=9, shift=3, downsample=False, F=999, f=81):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=2*dim, out_features=dim, act_layer=act_layer, drop=drop)
        # self.block_ref_3D = nn.Sequential(
        #     ShiftBlock(dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                drop=drop, attn_drop=attn_drop,
        #                drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer, M=M,
        #                shift=shift, downsample=False),
        #     norm_layer(dim),
        #     nn.Linear(dim, dim)
        # )
        #
        # self.block_ref_2D = nn.Sequential(
        #     ShiftBlock(dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                                drop=drop, attn_drop=attn_drop,
        #                                drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer, M=M,
        #                                shift=shift, downsample=False),
        # )
        #
        # self.block_src_2D = nn.Sequential(
        #     ShiftBlock(dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                                drop=drop, attn_drop=attn_drop,
        #                                drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer, M=M,
        #                                shift=shift, downsample=False),
        # )

        self.block = nn.Sequential(
            ShiftBlock(dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       drop=drop, attn_drop=attn_drop,
                                       drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer, M=M,
                                       shift=shift, downsample=False),
        )

        self.norm = norm_layer(dim)


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)


        self.M = M
        self.shift = shift
        self.downsample = downsample
        self.B = nn.Parameter(torch.randn(1, num_heads, f, F), requires_grad=True)
        self.get_q = nn.Sequential(norm_layer(dim), nn.Linear(dim, dim))
        self.get_k = nn.Sequential(norm_layer(dim), nn.Linear(dim, dim))
        self.get_v = nn.Sequential(norm_layer(dim), nn.Linear(dim, dim))

    def forward(self, ref_3D, ref_2D, src_2D):
        # ref_3D: (1, F, c) ref_2D: (1, F, c) src_2D: (b, f, c)
        _, F, _ = ref_3D.shape
        b, f, c = src_2D.shape
        ref = torch.cat([ref_3D, ref_2D], 0)
        ref = self.block(ref)
        v = self.get_v(ref[:1, :, :])
        k = self.get_k(ref[1:, :, :])
        # v = self.block(ref_3D)
        # v = self.get_v(v)
        # k = self.block(ref_2D)
        # k = self.get_k(k)
        src_2D = self.block(src_2D)
        q = self.get_q(src_2D)


        v = v.view(F, self.num_heads, c // self.num_heads).permute(1, 0, 2)
        k = k.view(F, self.num_heads, c // self.num_heads).permute(1, 0, 2)
        q = q.view(b, f, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)

        attn = torch.einsum('bhfc, hcp -> bhfp', q, k.transpose(-2, -1)) * self.scale + self.B
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        xx = torch.einsum('bhfp, hpc -> bhfc', attn, v)
        xx = xx.transpose(1, 2).reshape(b, f, c)
        xx = self.proj(xx)
        xx = self.proj_drop(xx)

        x = src_2D + self.drop_path(xx)
        x = x + self.drop_path(self.mlp(self.norm(x)))

        # x = torch.cat([src_2D, xx], -1)
        # x = self.mlp(self.norm(x))

        if self.downsample:
            x = x.view(b, -1, 3, c).mean(2)

        return v.transpose(0, 1).view(1, F, c), k.transpose(0, 1).view(1, F, c), x



class MutualBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, F=999, f=81):
        super().__init__()

        # self.num_heads = num_heads
        # head_dim = dim // num_heads
        # self.scale = qk_scale or head_dim ** -0.5
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=2*dim, hidden_features=2*dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.block_ref_3D = nn.Sequential(
            ShiftBlock(dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                       drop=drop, attn_drop=attn_drop,
                       drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer),
            norm_layer(dim),
            nn.Linear(dim, dim)
        )
        #
        self.block_ref_2D = nn.Sequential(
            ShiftBlock(dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       drop=drop, attn_drop=attn_drop,
                                       drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer),
            norm_layer(dim),
            nn.Linear(dim, dim)
        )

        self.block_src_2D = nn.Sequential(
            Block(dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       drop=drop, attn_drop=attn_drop,
                                       drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer, attn_type=RelativeAttention),
        )

        # self.norm = norm_layer(2*dim)
        #
        #
        # self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(drop)
        #
        # self.B = nn.Parameter(torch.randn(1, num_heads, f, F), requires_grad=True)
        # self.get_q = nn.Sequential(norm_layer(dim), nn.Linear(dim, dim))


    def forward(self, ref_3D, ref_2D, src_2D):
        # ref_3D: (1, F, c) ref_2D: (1, F, c) src_2D: (b, f, c)
        # _, F, _ = ref_3D.shape
        # b, f, c = src_2D.shape
        # v = self.block_ref_3D(ref_3D)
        # k = self.block_ref_2D(ref_2D)
        x = self.block_src_2D(src_2D)

        # q = self.get_q(src_2D)
        # v = v.view(F, self.num_heads, c // self.num_heads).permute(1, 0, 2)
        # k = k.view(F, self.num_heads, c // self.num_heads).permute(1, 0, 2)
        # q = q.view(b, f, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        #
        # attn = torch.einsum('bhfc, hcp -> bhfp', q, k.transpose(-2, -1)) * self.scale + self.B
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        #
        # xx = torch.einsum('bhfp, hpc -> bhfc', attn, v)
        # xx = xx.transpose(1, 2).reshape(b, f, c)
        # xx = self.proj(xx)
        # xx = self.proj_drop(xx)
        #
        # # x = src_2D + 0.0 * self.drop_path(xx)
        # # x = x + self.drop_path(self.mlp(self.norm(x)))
        #
        # x = torch.cat([src_2D, xx], -1)
        # x = self.mlp(self.norm(x))

        return None, None, x



class ShiftTransformer(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None, M=9, shift=3):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints  #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = num_joints * 3  #### output dimension is num_joints * 3

        ### spatial patch embedding
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        down_blocks = (np.log(num_frame // 9) / np.log(3)).astype(np.int32)

        self.blocks = nn.ModuleList([
            ShiftBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                M=M, shift=shift, downsample=True)
            for i in range(down_blocks)])

        if down_blocks < depth:
            self.blocks.extend(
                nn.ModuleList([
                    Block(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                    for i in range(down_blocks, depth)])
            )


        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)


        self.weighted_mean = torch.nn.Conv1d(in_channels=9, out_channels=1, kernel_size=1)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

    def Spatial_forward_features(self, x):
        b, _, f, p = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        x = rearrange(x, 'b c f p  -> (b f) p  c', )

        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)

        for blk in self.Spatial_blocks:
            x = blk(x)

        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) w c -> b f (w c)', f=f)
        return x

    def forward_features(self, x):
        b = x.shape[0]
        for blk in self.blocks:
            x = blk(x)

        x = self.Temporal_norm(x)
        ##### x size [b, f, emb_dim], then take weighted mean on frame dimension, we only predict 3D pose of the center frame
        x = self.weighted_mean(x)
        x = x.view(b, 1, -1)
        return x

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        b, _, _, p = x.shape
        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data
        x = self.Spatial_forward_features(x)
        x = self.forward_features(x)
        x = self.head(x)
        x = x.view(b, 1, p, -1)

        return x

class SwinTransformer(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None, M=9, dilated_M=9):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints  #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = num_joints * 3  #### output dimension is num_joints * 3

        ### spatial patch embedding
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        down_blocks = (np.log(num_frame // 9) / np.log(3)).astype(np.int32)

        self.blocks = nn.ModuleList([
            SwinBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                M=M, dilated_M=dilated_M, downsample=True)
            for i in range(down_blocks)])

        if down_blocks < depth:
            self.blocks.extend(
                nn.ModuleList([
                    SwinBlock(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                        M=M, dilated_M=dilated_M, downsample=False)
                    for i in range(down_blocks, depth)])
            )


        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)


        self.weighted_mean = torch.nn.Conv1d(in_channels=9, out_channels=1, kernel_size=1)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

    def Spatial_forward_features(self, x):
        b, _, f, p = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        x = rearrange(x, 'b c f p  -> (b f) p  c', )

        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)

        for blk in self.Spatial_blocks:
            x = blk(x)

        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) w c -> b f (w c)', f=f)
        return x

    def forward_features(self, x):
        b = x.shape[0]
        for blk in self.blocks:
            x = blk(x)

        x = self.Temporal_norm(x)
        ##### x size [b, f, emb_dim], then take weighted mean on frame dimension, we only predict 3D pose of the center frame
        x = self.weighted_mean(x)
        x = x.view(b, 1, -1)
        return x

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        b, _, _, p = x.shape
        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data
        x = self.Spatial_forward_features(x)
        x = self.forward_features(x)
        x = self.head(x)
        x = x.view(b, 1, p, -1)

        return x



class MutualTransformer(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None, M=9, shift=3, F=999):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints  #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = num_joints * 3  #### output dimension is num_joints * 3

        ### spatial patch embedding
        self.patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.patch_to_embedding_3D = nn.Linear(3, embed_dim_ratio)

        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(2)])

        down_blocks = (np.log(num_frame // 9) / np.log(3)).astype(np.int32)

        self.blocks = nn.ModuleList([
            ShiftMutualBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                M=M, shift=shift, downsample=True, F=F, f=num_frame//(3 ** i))
            for i in range(down_blocks)])

        if down_blocks < depth:
            self.blocks.extend(
                nn.ModuleList([
                    ShiftMutualBlock(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, M=M, shift=shift, downsample=False, F=F, f=9)
                    for i in range(down_blocks, depth)])
            )


        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)


        self.weighted_mean = torch.nn.Conv1d(in_channels=9, out_channels=1, kernel_size=1)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

    def Spatial_forward_features(self, x):
        b, f, p, _ = x.shape  ##### b is batch size
        x = rearrange(x, 'b f p c -> (b f) p c')
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        for blk in self.Spatial_blocks:
            x = blk(x)
        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) p c -> b f (p c)', f=f)
        # if not self.ST:
        #     x = x.mean(1)
        return x

    def forward_features(self, ref_3D, ref_2D, src_2D):
        b, _, f, p = src_2D.shape
        F = ref_3D.shape[2]
        src_2D = rearrange(src_2D, 'b c f p -> (b f) p c')
        src_2D = self.patch_to_embedding(src_2D)
        src_2D = rearrange(src_2D, '(b f) p c -> b f (p c)', f=f)
        ref_2D = rearrange(ref_2D, 'b c f p -> (b f) p c')
        ref_2D = self.patch_to_embedding(ref_2D)
        ref_2D = rearrange(ref_2D, '(b f) p c -> b f (p c)', f=F)
        ref_3D = rearrange(ref_3D, 'b c f p -> (b f) p c')
        ref_3D = self.patch_to_embedding_3D(ref_3D)
        ref_3D = rearrange(ref_3D, '(b f) p c -> b f (p c)', f=F)

        for blk in self.blocks:
            ref_3D, ref_2D, src_2D = blk(ref_3D, ref_2D, src_2D)

        x = self.Temporal_norm(src_2D)
        ##### x size [b, f, emb_dim], then take weighted mean on frame dimension, we only predict 3D pose of the center frame
        x = self.weighted_mean(x)
        x = x.view(b, 1, p, -1)
        return x

    def forward(self, ref_3D, ref_2D, src_2D):
        # ref_3D: (1, F, 17, 3) ref_2D: (1, F, 17, 2) src_2D: (b, f, 17, 2)
        ref_3D = ref_3D.mean(0, keepdim=True)
        ref_2D = ref_2D.mean(0, keepdim=True)
        ref_3D = ref_3D.permute(0, 3, 1, 2)
        ref_2D = ref_2D.permute(0, 3, 1, 2)
        src_2D = src_2D.permute(0, 3, 1, 2)
        b, _, f, p = src_2D.shape
        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data
        x = self.forward_features(ref_3D, ref_2D, src_2D)
        x = self.Spatial_forward_features(x)
        x = self.head(x)
        x = x.view(b, 1, p, -1)

        return x



class ShiftGroupDualTransformer(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None, merge=False, merge_type='mean', M=9, shift=3, F=999):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            merge (bool): merge anchors after each temporal block
            merge_type (str): 'mean' or 'transformer'
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints  #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = num_joints * 3  #### output dimension is num_joints * 3

        self.groups = ((1,2,3),(4,5,6),(0,7,8,9,10),(11,12,13),(14,15,16))
        self.anchors = (0,0,0,0,0)

        self.groups_anc = []
        if len(self.groups) > 0:
            self.inverse_group = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            count = 0
            for i in range(len(self.groups)):
                self.groups_anc.append(
                    torch.cat([torch.tensor([self.anchors[i]]).type(torch.long).cuda(non_blocking=True),
                               torch.tensor(self.groups[i]).type(torch.long).cuda(non_blocking=True)], dim=0))

                for g_idx in self.groups[i]:
                    self.inverse_group[g_idx] = count
                    count += 1
            self.inverse_group = torch.tensor(self.inverse_group).type(torch.long).cuda(non_blocking=True)

        self.anchor_list = torch.tensor((0, 4, 8, 14, 18), dtype=torch.long, device='cuda')
        self.other_list = torch.tensor((-1, 1, 2, 3, -1, 5, 6, 7, -1, 9, 10, 11, 12, 13, -1, 15, 16, 17, -1, 19, 20, 21),
                                  dtype=torch.long, device='cuda')

        self.group_list = [0, 4, 8, 14, 18, 22]


        ### spatial patch embedding
        self.patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.patch_to_embedding_3D = nn.Linear(3, embed_dim_ratio)

        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))


        # self.Temporal_pos_embed = [self.Temporal_pos_embed_0, self.Temporal_pos_embed_1, self.Temporal_pos_embed_2, self.Temporal_pos_embed_3, self.Temporal_pos_embed_4]
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(2)])

        down_blocks = (np.log(num_frame // 9) / np.log(3)).astype(np.int32)
        down_sample = [False, False, False, False]

        for i in range(down_blocks):
            down_sample[i] = True


        self.blocks = nn.ModuleList([
            nn.ModuleList([
                ShiftDualBlock(
                    dim=embed_dim_ratio * (len(g) + 1), num_heads=num_heads,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    M=M, shift=shift, downsample=down_sample[i])
                for g in self.groups
            ])
            for i in range(2)])
        if 2 < depth:
            self.blocks.extend(nn.ModuleList([
                nn.ModuleList([
                   DualBlock(
                        dim=embed_dim_ratio * (len(g) + 1), num_heads=num_heads,
                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, F=F, f=9)
                    for g in self.groups
                ])
                for i in range(2, depth)]))

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = nn.ModuleList([
            norm_layer(embed_dim_ratio * (len(g) + 1))
            for g in self.groups
        ])


        self.weighted_mean = nn.ModuleList([torch.nn.Conv1d(in_channels=9, out_channels=1, kernel_size=1)
                                            for g in self.groups])

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

        self.Merge = merge
        self.merge_type = merge_type
        if merge_type != 'mean':
            self.merge_blocks = nn.ModuleList([
                Block(
                    dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                for i in range(depth)
            ])
            # self.merge_norm = nn.ModuleList([
            #     norm_layer(embed_dim_ratio)
            #     for i in range(depth-1)
            # ])

    def split(self, x):
        x_g = []
        for i in range(len(self.groups_anc)):
            x_tmp = x[:, :, self.groups_anc[i], :]
            x_g.append(x_tmp.view(*x_tmp.shape[:2], -1))
        return x_g

    def split_efficient(self, x):
        x_g = []
        for i in range(len(self.groups_anc)):
            x_tmp = x[:, :, self.groups_anc[i], :]
            x_g.append(x_tmp)
        x_g = torch.cat(x_g, -2)
        return x_g


    def merge_efficient(self, x_b, b_i):
        ## x_b (b,f,4+4+6+4+4, 32)

        x_anchor = x_b[:, :, self.anchor_list, :]
        x_anchor = torch.mean(x_anchor, -2, keepdim=True)
        x_b = torch.cat([x_b, x_anchor], -2)
        x_b = x_b[:, :, self.other_list, :]
        return x_b

    def merge(self, x_b, b_i):
        b, f, _ = x_b[0].shape
        x_g = []
        anchor_0 = []
        anchor_8 = []

        for i in range(len(self.groups_anc)):
            p = self.groups_anc[i].shape[0] - 1
            if self.anchors[i] == 0:
                anchor_0.append(x_b[i].clone().view(b, f, p+1, -1)[:, :, 0:1, :])
            else:
                anchor_8.append(x_b[i].clone().view(b, f, p+1, -1)[:, :, 0:1, :])
        anchor_8.append(x_b[2].clone().view(b, f, 6, -1)[:, :, 3:4, :])
        anchor_8 = torch.stack(anchor_8, 0).mean(0)
        if self.merge_type == 'mean':
            anchor_0 = torch.stack(anchor_0, 0).mean(0)
        else:
            anchor_0 = torch.cat(anchor_0, 2).view(b*f, len(self.groups_anc), -1)
            anchor_0 = self.merge_blocks[b_i](anchor_0)
            anchor_0 = anchor_0.view(b, f, len(self.groups_anc), -1)

        for i in range(len(self.groups_anc)):
            p = self.groups_anc[i].shape[0] - 1
            if self.anchors[i] == 0:
                if anchor_0.shape[2] == 1:
                    x_tmp = torch.cat([anchor_0, x_b[i].clone().view(b, f, p + 1, -1)[:, :, 1:, :]], dim=2)
                else:
                    x_tmp = torch.cat([anchor_0[:, :, i:i+1, :], x_b[i].clone().view(b, f, p + 1, -1)[:, :, 1:, :]], dim=2)
            else:
                x_tmp = torch.cat([anchor_8, x_b[i].clone().view(b, f, p + 1, -1)[:, :, 1:, :]], dim=2)
            x_g.append(x_tmp.view(b, f, -1))
        return x_g


    def Spatial_forward_features(self, x):
        b, f, p, _ = x.shape  ##### b is batch size
        x = rearrange(x, 'b f p c -> (b f) p c')
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        for blk in self.Spatial_blocks:
            x = blk(x)
        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) p c -> b f (p c)', f=f)
        # if not self.ST:
        #     x = x.mean(1)
        return x


    def Temporal_efficient(self, ref_2D, x):
        b, _, f, p = x.shape
        x = rearrange(x, 'b c f p -> (b f) p c')
        x = self.patch_to_embedding(x)
        x = rearrange(x, '(b f) p c -> b f p c', f=f)
        x_g = self.split_efficient(x)
        F = ref_2D.shape[2]
        ref_2D = rearrange(ref_2D, 'b c f p -> (b f) p c')
        ref_2D = self.patch_to_embedding(ref_2D)
        ref_2D = rearrange(ref_2D, '(b f) p c -> b f p c', f=F)

        ref_2D_g = self.split_efficient(ref_2D)


        for blk_i in range(len(self.blocks)):
            blk = self.blocks[blk_i]
            x_b = []
            ref_2D_b = []
            for i in range(len(self.groups)):
                x_i = x_g[:, :, self.group_list[i]:self.group_list[i+1], :]
                x_i = x_i.view(b, x_i.shape[1], -1)
                ref_2D_i = ref_2D_g[:, :, self.group_list[i]:self.group_list[i+1], :]
                ref_2D_i = ref_2D_i.view(1, ref_2D_i.shape[1], -1)
                ref_2D_i, x_i = blk[i](ref_2D_i, x_i)

                if blk_i == len(self.blocks) - 1:
                    x_i = self.Temporal_norm[i](x_i)
                    x_i = self.weighted_mean[i](x_i).view(b, 1, len(self.groups[i]) + 1, -1)
                    x_i = x_i[:, :, 1:, :]
                    ref_2D_i = ref_2D_i.view(1, ref_2D_i.shape[1], len(self.groups[i]) + 1, -1)
                else:
                    x_i = x_i.view(b, x_i.shape[1], len(self.groups[i]) + 1, -1)
                    ref_2D_i = ref_2D_i.view(1, ref_2D_i.shape[1], len(self.groups[i]) + 1, -1)

                x_b.append(x_i)
                ref_2D_b.append(ref_2D_i)

            x_b = torch.cat(x_b, -2)
            ref_2D_b = torch.cat(ref_2D_b, -2)


            if self.Merge and blk_i < len(self.blocks) - 1:
                x_g = self.merge_efficient(x_b, blk_i)
                ref_2D_g = self.merge_efficient(ref_2D_b, blk_i)
            else:
                x_g = x_b.clone()

        x_out = x_g[:, :, self.inverse_group, :]
        # x_out = x_out.view(b, -1)
        # x_out = self.transform(x_out)
        x_out = x_out.view(b, 1, p, -1)

        return x_out


    def Temporal_forward_features(self, x):
        b, _, f, p = x.shape
        x = rearrange(x, 'b c f p -> (b f) p c')
        x = self.patch_to_embedding(x)
        x = rearrange(x, '(b f) p c -> b f p c', f=f)
        x_g = self.split(x)

        for blk_i in range(len(self.blocks)):
            blk = self.blocks[blk_i]
            x_b = []
            for i in range(len(self.groups)):
                x_i = x_g[i]
                x_i = blk[i](x_i)
                x_b.append(x_i)
            if self.Merge and blk_i < len(self.blocks):
                x_g = self.merge(x_b, blk_i)
            else:
                x_g = []
                for i in range(len(self.groups)):
                    x_g.append(x_b[i])
        x_out = []
        for i in range(len(self.groups)):
            x_i = x_g[i]
            x_i = self.Temporal_norm[i](x_i)  # (b, f', (p' c))
            x_i = self.weighted_mean[i](x_i).view(b, len(self.groups[i])+1, -1)
            x_i = x_i[:, 1:, :]
            x_out.append(x_i)

        x_out = torch.cat(x_out, 1)
        x_out = x_out[:, self.inverse_group, :]
        # x_out = x_out.view(b, -1)
        # x_out = self.transform(x_out)
        x_out = x_out.view(b, 1, p, -1)

        return x_out

    def forward(self, ref_2D, x):
        x = x.permute(0, 3, 1, 2)
        ref_2D = ref_2D.mean(0, keepdim=True)
        ref_2D = ref_2D.permute(0, 3, 1, 2)

        b, _, _, p = x.shape

        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data
        x = self.Temporal_efficient(ref_2D, x)
        x = self.Spatial_forward_features(x)

        x = self.head(x)
        x = x.view(b, 1, p, -1)
        # x[:, 0, (11, 12, 13, 14, 15, 16), :] = x[:, 0, (11, 12, 13, 14, 15, 16), :] + x[:, 0, 8:9, :]
        return x


class ShiftGroupDualShortTransformer(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None, merge=False, merge_type='mean', M=9, shift=3, F=999):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            merge (bool): merge anchors after each temporal block
            merge_type (str): 'mean' or 'transformer'
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints  #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = num_joints * 3  #### output dimension is num_joints * 3

        self.groups = ((1,2,3),(4,5,6),(0,7,8,9,10),(11,12,13),(14,15,16))
        self.anchors = (0,0,0,0,0)

        self.groups_anc = []
        if len(self.groups) > 0:
            self.inverse_group = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            count = 0
            for i in range(len(self.groups)):
                self.groups_anc.append(
                    torch.cat([torch.tensor([self.anchors[i]]).type(torch.long).cuda(non_blocking=True),
                               torch.tensor(self.groups[i]).type(torch.long).cuda(non_blocking=True)], dim=0))

                for g_idx in self.groups[i]:
                    self.inverse_group[g_idx] = count
                    count += 1
            self.inverse_group = torch.tensor(self.inverse_group).type(torch.long).cuda(non_blocking=True)

        self.anchor_list = torch.tensor((0, 4, 8, 14, 18), dtype=torch.long, device='cuda')
        self.other_list = torch.tensor((-1, 1, 2, 3, -1, 5, 6, 7, -1, 9, 10, 11, 12, 13, -1, 15, 16, 17, -1, 19, 20, 21),
                                  dtype=torch.long, device='cuda')

        self.group_list = [0, 4, 8, 14, 18, 22]


        ### spatial patch embedding
        self.patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.patch_to_embedding_3D = nn.Linear(3, embed_dim_ratio)

        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))


        # self.Temporal_pos_embed = [self.Temporal_pos_embed_0, self.Temporal_pos_embed_1, self.Temporal_pos_embed_2, self.Temporal_pos_embed_3, self.Temporal_pos_embed_4]
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(2)])

        down_blocks = (np.log(num_frame // 9) / np.log(3)).astype(np.int32)

        down_sample = [False, False, False, False]

        for i in range(down_blocks):
            down_sample[i] = True

        self.blocks = nn.ModuleList([
            nn.ModuleList([
                ShiftDualShortBlock(
                    dim=embed_dim_ratio * (len(g) + 1), num_heads=num_heads,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    M=M, shift=shift, downsample=down_sample[i])
                for g in self.groups
            ])
            for i in range(2)])
        if 2 < depth:
            self.blocks.extend(nn.ModuleList([
                nn.ModuleList([
                   DualShortBlock(
                        dim=embed_dim_ratio * (len(g) + 1), num_heads=num_heads,
                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, f=9)
                    for g in self.groups
                ])
                for i in range(2, depth)]))

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = nn.ModuleList([
            norm_layer(embed_dim_ratio * (len(g) + 1))
            for g in self.groups
        ])


        self.weighted_mean = nn.ModuleList([torch.nn.Conv1d(in_channels=9, out_channels=1, kernel_size=1)
                                            for g in self.groups])

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

        self.Merge = merge
        self.merge_type = merge_type
        if merge_type != 'mean':
            self.merge_blocks = nn.ModuleList([
                Block(
                    dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                for i in range(depth)
            ])
            # self.merge_norm = nn.ModuleList([
            #     norm_layer(embed_dim_ratio)
            #     for i in range(depth-1)
            # ])

    def split(self, x):
        x_g = []
        for i in range(len(self.groups_anc)):
            x_tmp = x[:, :, self.groups_anc[i], :]
            x_g.append(x_tmp.view(*x_tmp.shape[:2], -1))
        return x_g

    def split_efficient(self, x):
        x_g = []
        for i in range(len(self.groups_anc)):
            x_tmp = x[:, :, self.groups_anc[i], :]
            x_g.append(x_tmp)
        x_g = torch.cat(x_g, -2)
        return x_g


    def merge_efficient(self, x_b, b_i):
        ## x_b (b,f,4+4+6+4+4, 32)

        x_anchor = x_b[:, :, self.anchor_list, :]
        x_anchor = torch.mean(x_anchor, -2, keepdim=True)
        x_b = torch.cat([x_b, x_anchor], -2)
        x_b = x_b[:, :, self.other_list, :]
        return x_b

    def merge(self, x_b, b_i):
        b, f, _ = x_b[0].shape
        x_g = []
        anchor_0 = []
        anchor_8 = []

        for i in range(len(self.groups_anc)):
            p = self.groups_anc[i].shape[0] - 1
            if self.anchors[i] == 0:
                anchor_0.append(x_b[i].clone().view(b, f, p+1, -1)[:, :, 0:1, :])
            else:
                anchor_8.append(x_b[i].clone().view(b, f, p+1, -1)[:, :, 0:1, :])
        anchor_8.append(x_b[2].clone().view(b, f, 6, -1)[:, :, 3:4, :])
        anchor_8 = torch.stack(anchor_8, 0).mean(0)
        if self.merge_type == 'mean':
            anchor_0 = torch.stack(anchor_0, 0).mean(0)
        else:
            anchor_0 = torch.cat(anchor_0, 2).view(b*f, len(self.groups_anc), -1)
            anchor_0 = self.merge_blocks[b_i](anchor_0)
            anchor_0 = anchor_0.view(b, f, len(self.groups_anc), -1)

        for i in range(len(self.groups_anc)):
            p = self.groups_anc[i].shape[0] - 1
            if self.anchors[i] == 0:
                if anchor_0.shape[2] == 1:
                    x_tmp = torch.cat([anchor_0, x_b[i].clone().view(b, f, p + 1, -1)[:, :, 1:, :]], dim=2)
                else:
                    x_tmp = torch.cat([anchor_0[:, :, i:i+1, :], x_b[i].clone().view(b, f, p + 1, -1)[:, :, 1:, :]], dim=2)
            else:
                x_tmp = torch.cat([anchor_8, x_b[i].clone().view(b, f, p + 1, -1)[:, :, 1:, :]], dim=2)
            x_g.append(x_tmp.view(b, f, -1))
        return x_g


    def Spatial_forward_features(self, x):
        b, f, p, _ = x.shape  ##### b is batch size
        x = rearrange(x, 'b f p c -> (b f) p c')
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        for blk in self.Spatial_blocks:
            x = blk(x)
        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) p c -> b f (p c)', f=f)
        # if not self.ST:
        #     x = x.mean(1)
        return x


    def Temporal_efficient(self, ref_2D, x):
        b, _, f, p = x.shape

        x = rearrange(x, 'b c f p -> (b f) p c')
        x = self.patch_to_embedding(x)
        x = rearrange(x, '(b f) p c -> b f p c', f=f)
        x_g = self.split_efficient(x)
        ref_2D = rearrange(ref_2D, 'b c f p -> (b f) p c')
        ref_2D = self.patch_to_embedding(ref_2D)
        ref_2D = rearrange(ref_2D, '(b f) p c -> b f p c', f=f)

        ref_2D_g = self.split_efficient(ref_2D)


        for blk_i in range(len(self.blocks)):
            blk = self.blocks[blk_i]
            x_b = []
            ref_2D_b = []
            for i in range(len(self.groups)):
                x_i = x_g[:, :, self.group_list[i]:self.group_list[i+1], :]
                x_i = x_i.view(b, x_i.shape[1], -1)
                ref_2D_i = ref_2D_g[:, :, self.group_list[i]:self.group_list[i+1], :]
                ref_2D_i = ref_2D_i.view(b, ref_2D_i.shape[1], -1)
                ref_2D_i, x_i = blk[i](ref_2D_i, x_i)

                if blk_i == len(self.blocks) - 1:
                    x_i = self.Temporal_norm[i](x_i)
                    x_i = self.weighted_mean[i](x_i).view(b, 1, len(self.groups[i]) + 1, -1)
                    x_i = x_i[:, :, 1:, :]
                    ref_2D_i = self.Temporal_norm[i](ref_2D_i)
                    ref_2D_i = self.weighted_mean[i](ref_2D_i).view(b, 1, len(self.groups[i]) + 1, -1)
                    ref_2D_i = ref_2D_i[:, :, 1:, :]
                else:
                    x_i = x_i.view(b, x_i.shape[1], len(self.groups[i]) + 1, -1)
                    ref_2D_i = ref_2D_i.view(b, ref_2D_i.shape[1], len(self.groups[i]) + 1, -1)

                x_b.append(x_i)
                ref_2D_b.append(ref_2D_i)

            x_b = torch.cat(x_b, -2)
            ref_2D_b = torch.cat(ref_2D_b, -2)


            if self.Merge and blk_i < len(self.blocks) - 1:
                x_g = self.merge_efficient(x_b, blk_i)
                ref_2D_g = self.merge_efficient(ref_2D_b, blk_i)
            else:
                x_g = x_b.clone()
                ref_2D_g = ref_2D_b.clone()

        x_out = x_g[:, :, self.inverse_group, :]
        ref_2D_out = ref_2D_g[:, :, self.inverse_group, :]
        # x_out = x_out.view(b, -1)
        # x_out = self.transform(x_out)
        x_out = torch.cat([ref_2D_out, x_out], 0)
        x_out = x_out.view(b*2, 1, p, -1)

        return x_out


    def Temporal_forward_features(self, x):
        b, _, f, p = x.shape
        x = rearrange(x, 'b c f p -> (b f) p c')
        x = self.patch_to_embedding(x)
        x = rearrange(x, '(b f) p c -> b f p c', f=f)
        x_g = self.split(x)

        for blk_i in range(len(self.blocks)):
            blk = self.blocks[blk_i]
            x_b = []
            for i in range(len(self.groups)):
                x_i = x_g[i]
                x_i = blk[i](x_i)
                x_b.append(x_i)
            if self.Merge and blk_i < len(self.blocks):
                x_g = self.merge(x_b, blk_i)
            else:
                x_g = []
                for i in range(len(self.groups)):
                    x_g.append(x_b[i])
        x_out = []
        for i in range(len(self.groups)):
            x_i = x_g[i]
            x_i = self.Temporal_norm[i](x_i)  # (b, f', (p' c))
            x_i = self.weighted_mean[i](x_i).view(b, len(self.groups[i])+1, -1)
            x_i = x_i[:, 1:, :]
            x_out.append(x_i)

        x_out = torch.cat(x_out, 1)
        x_out = x_out[:, self.inverse_group, :]
        # x_out = x_out.view(b, -1)
        # x_out = self.transform(x_out)
        x_out = x_out.view(b, 1, p, -1)

        return x_out

    def forward(self, ref_2D, x, eval=False):
        x = x.permute(0, 3, 1, 2)
        ref_2D = ref_2D.permute(0, 3, 1, 2)

        b, _, _, p = x.shape

        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data
        x = self.Temporal_efficient(ref_2D, x)

        x = self.Spatial_forward_features(x)

        x = self.head(x)
        x = x.view(2*b, 1, p, -1)
        if eval:
            x = x[b:, :, :, :]
        # x[:, 0, (11, 12, 13, 14, 15, 16), :] = x[:, 0, (11, 12, 13, 14, 15, 16), :] + x[:, 0, 8:9, :]
        return x


class ShiftGroupRelativeTransformer(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None, merge=False, merge_type='mean', M=9, shift=3, F=999):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            merge (bool): merge anchors after each temporal block
            merge_type (str): 'mean' or 'transformer'
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints  #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = num_joints * 3  #### output dimension is num_joints * 3

        self.groups = ((1,2,3),(4,5,6),(0,7,8,9,10),(11,12,13),(14,15,16))
        self.anchors = (0,0,0,0,0)

        self.groups_anc = []
        if len(self.groups) > 0:
            self.inverse_group = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            count = 0
            for i in range(len(self.groups)):
                self.groups_anc.append(
                    torch.cat([torch.tensor([self.anchors[i]]).type(torch.long).cuda(non_blocking=True),
                               torch.tensor(self.groups[i]).type(torch.long).cuda(non_blocking=True)], dim=0))

                for g_idx in self.groups[i]:
                    self.inverse_group[g_idx] = count
                    count += 1
            self.inverse_group = torch.tensor(self.inverse_group).type(torch.long).cuda(non_blocking=True)

        self.anchor_list = torch.tensor((0, 4, 8, 14, 18), dtype=torch.long, device='cuda')
        self.other_list = torch.tensor((-1, 1, 2, 3, -1, 5, 6, 7, -1, 9, 10, 11, 12, 13, -1, 15, 16, 17, -1, 19, 20, 21),
                                  dtype=torch.long, device='cuda')

        self.group_list = [0, 4, 8, 14, 18, 22]


        ### spatial patch embedding
        self.patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.patch_to_embedding_3D = nn.Linear(3, embed_dim_ratio)

        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))


        # self.Temporal_pos_embed = [self.Temporal_pos_embed_0, self.Temporal_pos_embed_1, self.Temporal_pos_embed_2, self.Temporal_pos_embed_3, self.Temporal_pos_embed_4]
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(2)])

        down_blocks = (np.log(num_frame // 9) / np.log(3)).astype(np.int32)
        down_sample = [False, False, False, False]

        for i in range(down_blocks):
            down_sample[i] = True


        self.blocks = nn.ModuleList([
            nn.ModuleList([
                ShiftParallelBlock(
                    dim=embed_dim_ratio * (len(g) + 1), num_heads=num_heads,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    M=M, shift=shift, downsample=down_sample[i])
                for g in self.groups
            ])
            for i in range(2)])
        if 2 < depth:
            self.blocks.extend(nn.ModuleList([
                nn.ModuleList([
                   ParallelBlock(
                        dim=embed_dim_ratio * (len(g) + 1), num_heads=num_heads,
                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                    for g in self.groups
                ])
                for i in range(2, depth)]))

        self.relative = nn.ModuleList([
                   RelativeBlock(
                        dim=embed_dim_ratio * (len(g) + 1), num_heads=num_heads,
                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, f=9)
                    for g in self.groups
                ])


        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = nn.ModuleList([
            norm_layer(embed_dim_ratio * (len(g) + 1))
            for g in self.groups
        ])


        self.weighted_mean = nn.ModuleList([torch.nn.Conv1d(in_channels=9, out_channels=1, kernel_size=1)
                                            for g in self.groups])

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

        self.Merge = merge
        self.merge_type = merge_type
        if merge_type != 'mean':
            self.merge_blocks = nn.ModuleList([
                Block(
                    dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                for i in range(depth)
            ])
            # self.merge_norm = nn.ModuleList([
            #     norm_layer(embed_dim_ratio)
            #     for i in range(depth-1)
            # ])

    def split(self, x):
        x_g = []
        for i in range(len(self.groups_anc)):
            x_tmp = x[:, :, self.groups_anc[i], :]
            x_g.append(x_tmp.view(*x_tmp.shape[:2], -1))
        return x_g

    def split_efficient(self, x):
        x_g = []
        for i in range(len(self.groups_anc)):
            x_tmp = x[:, :, self.groups_anc[i], :]
            x_g.append(x_tmp)
        x_g = torch.cat(x_g, -2)
        return x_g


    def merge_efficient(self, x_b, b_i):
        ## x_b (b,f,4+4+6+4+4, 32)

        x_anchor = x_b[:, :, self.anchor_list, :]
        x_anchor = torch.mean(x_anchor, -2, keepdim=True)
        x_b = torch.cat([x_b, x_anchor], -2)
        x_b = x_b[:, :, self.other_list, :]
        return x_b

    def merge(self, x_b, b_i):
        b, f, _ = x_b[0].shape
        x_g = []
        anchor_0 = []
        anchor_8 = []

        for i in range(len(self.groups_anc)):
            p = self.groups_anc[i].shape[0] - 1
            if self.anchors[i] == 0:
                anchor_0.append(x_b[i].clone().view(b, f, p+1, -1)[:, :, 0:1, :])
            else:
                anchor_8.append(x_b[i].clone().view(b, f, p+1, -1)[:, :, 0:1, :])
        anchor_8.append(x_b[2].clone().view(b, f, 6, -1)[:, :, 3:4, :])
        anchor_8 = torch.stack(anchor_8, 0).mean(0)
        if self.merge_type == 'mean':
            anchor_0 = torch.stack(anchor_0, 0).mean(0)
        else:
            anchor_0 = torch.cat(anchor_0, 2).view(b*f, len(self.groups_anc), -1)
            anchor_0 = self.merge_blocks[b_i](anchor_0)
            anchor_0 = anchor_0.view(b, f, len(self.groups_anc), -1)

        for i in range(len(self.groups_anc)):
            p = self.groups_anc[i].shape[0] - 1
            if self.anchors[i] == 0:
                if anchor_0.shape[2] == 1:
                    x_tmp = torch.cat([anchor_0, x_b[i].clone().view(b, f, p + 1, -1)[:, :, 1:, :]], dim=2)
                else:
                    x_tmp = torch.cat([anchor_0[:, :, i:i+1, :], x_b[i].clone().view(b, f, p + 1, -1)[:, :, 1:, :]], dim=2)
            else:
                x_tmp = torch.cat([anchor_8, x_b[i].clone().view(b, f, p + 1, -1)[:, :, 1:, :]], dim=2)
            x_g.append(x_tmp.view(b, f, -1))
        return x_g


    def Spatial_forward_features(self, x):
        b, f, p, _ = x.shape  ##### b is batch size
        x = rearrange(x, 'b f p c -> (b f) p c')
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        for blk in self.Spatial_blocks:
            x = blk(x)
        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) p c -> b f (p c)', f=f)
        # if not self.ST:
        #     x = x.mean(1)
        return x


    def Temporal_efficient(self, ref_2D, x):
        b, _, f, p = x.shape

        x = rearrange(x, 'b c f p -> (b f) p c')
        x = self.patch_to_embedding(x)
        x = rearrange(x, '(b f) p c -> b f p c', f=f)
        x_g = self.split_efficient(x)

        ref_2D = rearrange(ref_2D, 'b c f p -> (b f) p c')
        ref_2D = self.patch_to_embedding(ref_2D)
        ref_2D = rearrange(ref_2D, '(b f) p c -> b f p c', f=f)
        ref_2D_g = self.split_efficient(ref_2D)


        for blk_i in range(len(self.blocks)):
            blk = self.blocks[blk_i]
            x_b = []
            ref_2D_b = []
            for i in range(len(self.groups)):
                x_i = x_g[:, :, self.group_list[i]:self.group_list[i+1], :]
                x_i = x_i.view(b, x_i.shape[1], -1)
                ref_2D_i = ref_2D_g[:, :, self.group_list[i]:self.group_list[i+1], :]
                ref_2D_i = ref_2D_i.view(b, ref_2D_i.shape[1], -1)

                ref_2D_i, x_i = blk[i](ref_2D_i, x_i)

                if blk_i == len(self.blocks) - 1:
                    ref_relative, x_relative = self.relative[i](ref_2D_i, x_i)
                    x_i = self.Temporal_norm[i](x_i)
                    x_i = self.weighted_mean[i](x_i).view(b, 1, len(self.groups[i]) + 1, -1)
                    x_i = x_i[:, :, 1:, :]
                    ref_2D_i = self.Temporal_norm[i](ref_2D_i)
                    ref_2D_i = self.weighted_mean[i](ref_2D_i).view(b, 1, len(self.groups[i]) + 1, -1)
                    ref_2D_i = ref_2D_i[:, :, 1:, :]
                else:
                    x_i = x_i.view(b, x_i.shape[1], len(self.groups[i]) + 1, -1)
                    ref_2D_i = ref_2D_i.view(b, ref_2D_i.shape[1], len(self.groups[i]) + 1, -1)

                x_b.append(x_i)
                ref_2D_b.append(ref_2D_i)

            x_b = torch.cat(x_b, -2)
            ref_2D_b = torch.cat(ref_2D_b, -2)


            if self.Merge and blk_i < len(self.blocks) - 1:
                x_g = self.merge_efficient(x_b, blk_i)
                ref_2D_g = self.merge_efficient(ref_2D_b, blk_i)
            else:
                x_g = x_b.clone()
                ref_2D_g = ref_2D_b.clone()

        x_out = x_g[:, :, self.inverse_group, :]
        ref_out = ref_2D_g[:, :, self.inverse_group, :]
        x_out = x_out.view(b, 1, p, -1)
        ref_out = ref_out.view(b, 1, p, -1)

        return x_out, ref_out, ref_relative, x_relative


    def Temporal_forward_features(self, x):
        b, _, f, p = x.shape
        x = rearrange(x, 'b c f p -> (b f) p c')
        x = self.patch_to_embedding(x)
        x = rearrange(x, '(b f) p c -> b f p c', f=f)
        x_g = self.split(x)

        for blk_i in range(len(self.blocks)):
            blk = self.blocks[blk_i]
            x_b = []
            for i in range(len(self.groups)):
                x_i = x_g[i]
                x_i = blk[i](x_i)
                x_b.append(x_i)
            if self.Merge and blk_i < len(self.blocks):
                x_g = self.merge(x_b, blk_i)
            else:
                x_g = []
                for i in range(len(self.groups)):
                    x_g.append(x_b[i])
        x_out = []
        for i in range(len(self.groups)):
            x_i = x_g[i]
            x_i = self.Temporal_norm[i](x_i)  # (b, f', (p' c))
            x_i = self.weighted_mean[i](x_i).view(b, len(self.groups[i])+1, -1)
            x_i = x_i[:, 1:, :]
            x_out.append(x_i)

        x_out = torch.cat(x_out, 1)
        x_out = x_out[:, self.inverse_group, :]
        # x_out = x_out.view(b, -1)
        # x_out = self.transform(x_out)
        x_out = x_out.view(b, 1, p, -1)

        return x_out

    def forward(self, ref_2D, x):
        x = x.permute(0, 3, 1, 2)
        ref_2D = ref_2D.mean(0, keepdim=True)
        ref_2D = ref_2D.permute(0, 3, 1, 2)

        b, _, _, p = x.shape

        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data
        x = self.Temporal_efficient(ref_2D, x)
        x = self.Spatial_forward_features(x)

        x = self.head(x)
        x = x.view(b, 1, p, -1)
        # x[:, 0, (11, 12, 13, 14, 15, 16), :] = x[:, 0, (11, 12, 13, 14, 15, 16), :] + x[:, 0, 8:9, :]
        return x


class ShiftGroupBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, M=9, shift=3, downsample=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GroupAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, seq=M)
        self.norm2 = norm_layer(dim)
        self.shift_attn1 = GroupAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, seq=M)
        self.norm3 = norm_layer(dim)
        self.shift_attn2 = GroupAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, seq=M)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm4 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm5 = norm_layer(dim)
        self.shift_mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm6 = norm_layer(dim)
        self.shift_mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.M = M
        self.shift = shift
        self.downsample = downsample


    def shift_features(self, x):
        # x: (b, f, c)
        x = torch.cat([x[:, self.shift:, :], x[:, :self.shift, :]], 1)
        mask = torch.zeros(self.M, self.M, device=x.device, dtype=x.dtype)
        mask[:self.M - self.shift, self.M - self.shift:] = -np.inf
        mask[self.M - self.shift:, :self.M - self.shift] = -np.inf
        group_num = x.shape[1] // self.M
        mask_zeros = torch.zeros(group_num - 1, self.M, self.M, device=x.device, dtype=x.dtype)
        mask = torch.cat([mask_zeros, mask.view(1, self.M, self.M)], 0)
        return x, mask


    def forward(self, x):
        # x: (b, f, c)
        b, f, c = x.shape

        x = x.view(b, -1, self.M, c).view(-1, self.M, c)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm4(x)))
        x = x.view(b, -1, self.M, c).view(b, f, c)

        if f > 9:
            x, mask = self.shift_features(x)
        else:
            mask = torch.zeros(1, self.M, self.M, device=x.device, dtype=x.dtype)

        x = x.view(b, -1, self.M, c).view(-1, self.M, c)
        mask = mask.repeat(b, 1, 1)
        x = x + self.drop_path(self.shift_attn1(self.norm2(x), mask))
        x = x + self.drop_path(self.shift_mlp1(self.norm5(x)))
        x = x.view(b, -1, self.M, c).view(b, f, c)

        if f > 9:
            x, mask = self.shift_features(x)
        else:
            mask = torch.zeros(1, self.M, self.M, device=x.device, dtype=x.dtype)

        x = x.view(b, -1, self.M, c).view(-1, self.M, c)
        mask = mask.repeat(b, 1, 1)
        x = x + self.drop_path(self.shift_attn2(self.norm3(x), mask))
        x = x + self.drop_path(self.shift_mlp2(self.norm6(x)))
        x = x.view(b, -1, self.M, c).view(b, f, c)

        if f > 9:
            x = torch.cat([x[:, -self.shift*2:, :], x[:, :-self.shift*2, :]], 1)


        if self.downsample:
            x = x.view(b, -1, 3, c).mean(2)
        return x

class GroupBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, downsample=False, seq=9):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GroupAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, seq=seq)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.downsample = downsample



    def forward(self, x):
        # x: (b, f, c)
        b, f, c = x.shape
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.downsample:
            x = x.view(b, -1, 3, c).mean(2)
        return x




class PoseTransformer(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None, input_norm=False,
                 attention_type='Attention', factor_q=1, factor_k=5):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints  #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = num_joints * 3  #### output dimension is num_joints * 3

        ### spatial patch embedding
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))

        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if attention_type == 'Attention':
            attn_type = Attention
        elif attention_type == 'Down':
            attn_type = DownSampleAttention
        else:
            attn_type = ProbSparseAttention

        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        if attention_type == 'Down':
            self.blocks = nn.ModuleList([
                TBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, down_factor=3)
                for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    attn_type=attn_type, factor_q=factor_q, factor_k=factor_k)
                for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        ####### A easy way to implement weighted mean
        if attention_type == 'Down':
            self.weighted_mean = torch.nn.Conv1d(in_channels=num_frame // (factor_q ** depth), out_channels=1,
                                                 kernel_size=1)
        else:
            self.weighted_mean = torch.nn.Conv1d(in_channels=num_frame, out_channels=1, kernel_size=1)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

        if input_norm:
            self.input_norm = nn.BatchNorm2d(2)
        else:
            self.input_norm = None

    def Spatial_forward_features(self, x):
        b, _, f, p = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        x = rearrange(x, 'b c f p  -> (b f) p  c', )

        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)

        for blk in self.Spatial_blocks:
            x = blk(x)

        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) w c -> b f (w c)', f=f)
        return x

    def forward_features(self, x):
        b = x.shape[0]
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)

        x = self.Temporal_norm(x)
        ##### x size [b, f, emb_dim], then take weighted mean on frame dimension, we only predict 3D pose of the center frame
        x = self.weighted_mean(x)
        x = x.view(b, 1, -1)
        return x

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        if self.input_norm is not None:
            x = self.input_norm(x)
        b, _, _, p = x.shape
        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data
        x = self.Spatial_forward_features(x)
        x = self.forward_features(x)
        x = self.head(x)

        x = x.view(b, 1, p, -1)

        return x


class PoseTransformerInverse(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None, down_factor=1):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints  #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = num_joints * 3  #### output dimension is num_joints * 3

        ### spatial patch embedding
        self.patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)

        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))

        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        down_blocks = (np.log(num_frame // 9) / np.log(3)).astype(np.int32)

        self.blocks = nn.ModuleList([
            TBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                down_factor=down_factor)
            for i in range(min(depth, down_blocks))])
        if down_blocks < depth:
            self.blocks.extend(nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                for i in range(down_blocks, depth)]))

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        self.split = nn.Linear(embed_dim, embed_dim)

        ####### A easy way to implement weighted mean
        # if attention_type == 'Down':
        #     self.weighted_mean = torch.nn.Conv1d(in_channels=num_frame // (factor_q ** depth), out_channels=1, kernel_size=1)
        # else:
        #     self.weighted_mean = torch.nn.Conv1d(in_channels=num_frame, out_channels=1, kernel_size=1)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim_ratio),
            nn.Linear(embed_dim_ratio, 3),
        )

    def Spatial_forward_features(self, x):
        b, p, _ = x.shape  ##### b is batch size
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)

        for blk in self.Spatial_blocks:
            x = blk(x)

        x = self.Spatial_norm(x)
        return x

    def Temporal_forward_features(self, x):
        b, _, f, p = x.shape
        x = rearrange(x, 'b c f p -> (b f) p c')
        x = self.patch_to_embedding(x)
        x = rearrange(x, '(b f) p c -> b f (p c)', f=f)
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)

        x = self.Temporal_norm(x)
        ##### x size [b, f, emb_dim], then take weighted mean on frame dimension, we only predict 3D pose of the center frame
        # x = self.weighted_mean(x)
        x = x.mean(1)
        x = x.view(b, -1)
        x = self.split(x)
        x = x.view(b, p, -1)
        return x

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)

        b, _, _, p = x.shape

        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data
        x = self.Temporal_forward_features(x)
        x = self.Spatial_forward_features(x)
        x = self.head(x)

        x = x.view(b, 1, p, -1)

        return x



class MEAN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # (batch, frames, ...)
        x = x.mean(1, keepdim=True)
        return x


class PoseGroupTransformer(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None, down_factor=1, merge=False, merge_type='mean', ST=False, pos_type='learnable', weighted=False):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            merge (bool): merge anchors after each temporal block
            merge_type (str): 'mean' or 'transformer'
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints  #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = num_joints * 3  #### output dimension is num_joints * 3

        self.groups = ((1,2,3),(4,5,6),(0,7,8,9,10),(11,12,13),(14,15,16))
        self.anchors = (0,0,0,0,0)

        self.groups_anc = []
        if len(self.groups) > 0:
            self.inverse_group = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            count = 0
            for i in range(len(self.groups)):
                self.groups_anc.append(
                    torch.cat([torch.tensor([self.anchors[i]]).type(torch.long).cuda(non_blocking=True),
                               torch.tensor(self.groups[i]).type(torch.long).cuda(non_blocking=True)], dim=0))

                for g_idx in self.groups[i]:
                    self.inverse_group[g_idx] = count
                    count += 1
            self.inverse_group = torch.tensor(self.inverse_group).type(torch.long).cuda(non_blocking=True)


        ### spatial patch embedding
        self.patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)

        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))

        if pos_type == 'learnable':
            self.Temporal_pos_embed_0 = nn.Parameter(torch.zeros(1, num_frame, embed_dim_ratio * (len(self.groups[0]) + 1)))
            self.Temporal_pos_embed_1 = nn.Parameter(torch.zeros(1, num_frame, embed_dim_ratio * (len(self.groups[1]) + 1)))
            self.Temporal_pos_embed_2 = nn.Parameter(torch.zeros(1, num_frame, embed_dim_ratio * (len(self.groups[2]) + 1)))
            self.Temporal_pos_embed_3 = nn.Parameter(torch.zeros(1, num_frame, embed_dim_ratio * (len(self.groups[3]) + 1)))
            self.Temporal_pos_embed_4 = nn.Parameter(torch.zeros(1, num_frame, embed_dim_ratio * (len(self.groups[4]) + 1)))
        else:
            self.Temporal_pos_embed_0 = nn.Parameter(self.get_pos_embed(num_frame, embed_dim_ratio * (len(self.groups[0]) + 1)), requires_grad=False)
            self.Temporal_pos_embed_1 = nn.Parameter(self.get_pos_embed(num_frame, embed_dim_ratio * (len(self.groups[1]) + 1)), requires_grad=False)
            self.Temporal_pos_embed_2 = nn.Parameter(self.get_pos_embed(num_frame, embed_dim_ratio * (len(self.groups[2]) + 1)), requires_grad=False)
            self.Temporal_pos_embed_3 = nn.Parameter(self.get_pos_embed(num_frame, embed_dim_ratio * (len(self.groups[3]) + 1)), requires_grad=False)
            self.Temporal_pos_embed_4 = nn.Parameter(self.get_pos_embed(num_frame, embed_dim_ratio * (len(self.groups[4]) + 1)), requires_grad=False)


        # self.Temporal_pos_embed = [self.Temporal_pos_embed_0, self.Temporal_pos_embed_1, self.Temporal_pos_embed_2, self.Temporal_pos_embed_3, self.Temporal_pos_embed_4]
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        down_blocks = (np.log(num_frame // 9) / np.log(3)).astype(np.int32)

        self.blocks = nn.ModuleList([
            nn.ModuleList([
                TBlock(
                    dim=embed_dim_ratio * (len(g) + 1), num_heads=num_heads,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    down_factor=down_factor)
                for g in self.groups
            ])
            for i in range(min(depth, down_blocks))])
        if down_blocks < depth:
            self.blocks.extend(nn.ModuleList([
                nn.ModuleList([
                    TBlock(
                        dim=embed_dim_ratio * (len(g) + 1), num_heads=num_heads,
                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                        down_factor=1)
                    for g in self.groups
                ])
                for i in range(down_blocks, depth)]))

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = nn.ModuleList([
            norm_layer(embed_dim_ratio * (len(g) + 1))
            for g in self.groups
        ])

        # self.transform = nn.Linear(embed_dim, embed_dim)

        ####### A easy way to implement weighted mean
        # if attention_type == 'Down':
        #     self.weighted_mean = torch.nn.Conv1d(in_channels=num_frame // (factor_q ** depth), out_channels=1, kernel_size=1)
        # else:
        if weighted:
            self.weighted_mean = nn.ModuleList([torch.nn.Conv1d(in_channels=num_frame, out_channels=1, kernel_size=1)
                                                for g in self.groups])
        else:
            self.weighted_mean = nn.ModuleList([MEAN() for g in self.groups])

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim_ratio),
            nn.Linear(embed_dim_ratio, 3),
        )
        self.int_head = nn.Sequential(
            nn.LayerNorm(embed_dim_ratio),
            nn.Linear(embed_dim_ratio, 3),
        )

        self.Merge = merge
        self.merge_type = merge_type
        if merge_type != 'mean':
            self.merge_blocks = nn.ModuleList([
                Block(
                    dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                for i in range(depth)
            ])
            # self.merge_norm = nn.ModuleList([
            #     norm_layer(embed_dim_ratio)
            #     for i in range(depth-1)
            # ])
        self.ST = ST

    def get_pos_embed(self, seq, dim):
        pad = seq // 2
        pos = torch.arange(-pad, pad+1, dtype=torch.float32)
        pos = pos.view(-1, 1).repeat(1, dim//2)
        div = torch.arange(0, dim//2, dtype=torch.float32)
        div = div.view(1, -1).repeat(seq, 1)
        div = 10000 ** (2.0 * div / dim)
        PE_sin = torch.sin(pos / div)
        PE_cos = torch.cos(pos / div)
        PE = torch.stack([PE_sin, PE_cos], 2).view(1, seq, dim)
        return PE



    def split(self, x):
        x_g = []
        for i in range(len(self.groups_anc)):
            x_tmp = x[:, :, self.groups_anc[i], :]
            x_g.append(x_tmp.view(*x_tmp.shape[:2], -1))
        return x_g

    def merge(self, x_b, b_i):
        b, f, _ = x_b[0].shape
        x_g = []
        anchor_0 = []
        anchor_8 = []

        for i in range(len(self.groups_anc)):
            p = self.groups_anc[i].shape[0] - 1
            if self.anchors[i] == 0:
                anchor_0.append(x_b[i].clone().view(b, f, p+1, -1)[:, :, 0:1, :])
            else:
                anchor_8.append(x_b[i].clone().view(b, f, p+1, -1)[:, :, 0:1, :])
        anchor_8.append(x_b[2].clone().view(b, f, 6, -1)[:, :, 3:4, :])
        anchor_8 = torch.stack(anchor_8, 0).mean(0)
        if self.merge_type == 'mean':
            anchor_0 = torch.stack(anchor_0, 0).mean(0)
        else:
            anchor_0 = torch.cat(anchor_0, 2).view(b*f, len(self.groups_anc), -1)
            anchor_0 = self.merge_blocks[b_i](anchor_0)
            anchor_0 = anchor_0.view(b, f, len(self.groups_anc), -1)

        for i in range(len(self.groups_anc)):
            p = self.groups_anc[i].shape[0] - 1
            if self.anchors[i] == 0:
                if anchor_0.shape[2] == 1:
                    x_tmp = torch.cat([anchor_0, x_b[i].clone().view(b, f, p + 1, -1)[:, :, 1:, :]], dim=2)
                else:
                    x_tmp = torch.cat([anchor_0[:, :, i:i+1, :], x_b[i].clone().view(b, f, p + 1, -1)[:, :, 1:, :]], dim=2)
            else:
                x_tmp = torch.cat([anchor_8, x_b[i].clone().view(b, f, p + 1, -1)[:, :, 1:, :]], dim=2)
            x_g.append(x_tmp.view(b, f, -1))
        return x_g




    def Spatial_forward_features(self, x):
        b, f, p, _ = x.shape  ##### b is batch size
        x = rearrange(x, 'b f p c -> (b f) p c')
        if self.ST:
            x = self.patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        for blk in self.Spatial_blocks:
            x = blk(x)
        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) p c -> b f p c', f=f)
        # if not self.ST:
        #     x = x.mean(1)
        return x

    def Temporal_forward_features(self, x):
        if self.ST:
            x = rearrange(x, 'b f p c -> b c f p')
        b, _, f, p = x.shape
        x = rearrange(x, 'b c f p -> (b f) p c')
        if not self.ST:
            x = self.patch_to_embedding(x)
        x = rearrange(x, '(b f) p c -> b f p c', f=f)
        x_g = self.split(x)
        x_b = []
        # for i in range(len(self.groups)):
        #     x_i = x_g[i]
        #     x_i += self.Temporal_pos_embed[i]
        #     x_i = self.pos_drop(x_i)
        #     x_b.append(x_i)

        x_0 = x_g[0]
        x_0 += self.Temporal_pos_embed_0
        x_0 = self.pos_drop(x_0)
        x_b.append(x_0)

        x_1 = x_g[1]
        x_1 += self.Temporal_pos_embed_1
        x_1 = self.pos_drop(x_1)
        x_b.append(x_1)

        x_2 = x_g[2]
        x_2 += self.Temporal_pos_embed_2
        x_2 = self.pos_drop(x_2)
        x_b.append(x_2)

        x_3 = x_g[3]
        x_3 += self.Temporal_pos_embed_3
        x_3 = self.pos_drop(x_3)
        x_b.append(x_3)

        x_4 = x_g[4]
        x_4 += self.Temporal_pos_embed_4
        x_4 = self.pos_drop(x_4)
        x_b.append(x_4)


        x_g = []
        for i in range(len(self.groups)):
            x_g.append(x_b[i])
        for blk_i in range(len(self.blocks)):
            blk = self.blocks[blk_i]
            x_b = []
            for i in range(len(self.groups)):
                x_i = x_g[i]
                x_i = blk[i](x_i)
                x_b.append(x_i)
            if self.Merge and blk_i < len(self.blocks):
                x_g = self.merge(x_b, blk_i)
            else:
                x_g = []
                for i in range(len(self.groups)):
                    x_g.append(x_b[i])
        x_out = []
        for i in range(len(self.groups)):
            x_i = x_g[i]
            x_i = self.Temporal_norm[i](x_i)  # (b, f', (p' c))
            x_i = self.weighted_mean[i](x_i).view(b, len(self.groups[i])+1, -1)
            # x_i = x_i.mean(1).view(b, len(self.groups[i])+1, -1)
            x_i = x_i[:, 1:, :]
            x_out.append(x_i)

        x_out = torch.cat(x_out, 1)
        x_out = x_out[:, self.inverse_group, :]
        # x_out = x_out.view(b, -1)
        # x_out = self.transform(x_out)
        x_out = x_out.view(b, 1, p, -1)

        return x_out

    def forward(self, x, int=False):
        x = x.permute(0, 3, 1, 2)

        b, _, _, p = x.shape

        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data
        if self.ST:
            x = rearrange(x, 'b c f p -> b f p c')
            x = self.Spatial_forward_features(x)
            x = self.Temporal_forward_features(x)
        else:
            x = self.Temporal_forward_features(x)
            x = self.Spatial_forward_features(x)

        x = self.head(x)
        x = x.view(b, 1, p, -1)
        # x[:, 0, (11, 12, 13, 14, 15, 16), :] = x[:, 0, (11, 12, 13, 14, 15, 16), :] + x[:, 0, 8:9, :]
        return x


class ShiftGroupTransformer(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None, merge=False, merge_type='mean', M=9, shift=3):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            merge (bool): merge anchors after each temporal block
            merge_type (str): 'mean' or 'transformer'
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints  #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = num_joints * 3  #### output dimension is num_joints * 3

        self.groups = ((1,2,3),(4,5,6),(0,7,8,9,10),(11,12,13),(14,15,16))
        self.anchors = (0,0,0,0,0)

        self.groups_anc = []
        if len(self.groups) > 0:
            self.inverse_group = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            count = 0
            for i in range(len(self.groups)):
                self.groups_anc.append(
                    torch.cat([torch.tensor([self.anchors[i]]).type(torch.long).cuda(non_blocking=True),
                               torch.tensor(self.groups[i]).type(torch.long).cuda(non_blocking=True)], dim=0))

                for g_idx in self.groups[i]:
                    self.inverse_group[g_idx] = count
                    count += 1
            self.inverse_group = torch.tensor(self.inverse_group).type(torch.long).cuda(non_blocking=True)

        self.anchor_list = torch.tensor((0, 4, 8, 14, 18), dtype=torch.long, device='cuda')
        self.other_list = torch.tensor((-1, 1, 2, 3, -1, 5, 6, 7, -1, 9, 10, 11, 12, 13, -1, 15, 16, 17, -1, 19, 20, 21),
                                  dtype=torch.long, device='cuda')

        self.group_list = [0, 4, 8, 14, 18, 22]


        ### spatial patch embedding
        self.patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)

        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))


        # self.Temporal_pos_embed = [self.Temporal_pos_embed_0, self.Temporal_pos_embed_1, self.Temporal_pos_embed_2, self.Temporal_pos_embed_3, self.Temporal_pos_embed_4]
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(2)])

        down_blocks = (np.log(num_frame // 9) / np.log(3)).astype(np.int32)

        self.blocks = nn.ModuleList([
            nn.ModuleList([
                ShiftBlock(
                    dim=embed_dim_ratio * (len(g) + 1), num_heads=num_heads,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    M=M, shift=shift, downsample=True)
                for g in self.groups
            ])
            for i in range(min(depth, down_blocks))])
        if down_blocks < depth:
            self.blocks.extend(nn.ModuleList([
                nn.ModuleList([
                    Block(
                        dim=embed_dim_ratio * (len(g) + 1), num_heads=num_heads,
                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                    for g in self.groups
                ])
                for i in range(down_blocks, depth)]))

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = nn.ModuleList([
            norm_layer(embed_dim_ratio * (len(g) + 1))
            for g in self.groups
        ])


        self.weighted_mean = nn.ModuleList([torch.nn.Conv1d(in_channels=9, out_channels=1, kernel_size=1)
                                            for g in self.groups])

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

        self.Merge = merge
        self.merge_type = merge_type
        if merge_type != 'mean':
            self.merge_blocks = nn.ModuleList([
                Block(
                    dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                for i in range(depth)
            ])
            # self.merge_norm = nn.ModuleList([
            #     norm_layer(embed_dim_ratio)
            #     for i in range(depth-1)
            # ])

    def split(self, x):
        x_g = []
        for i in range(len(self.groups_anc)):
            x_tmp = x[:, :, self.groups_anc[i], :]
            x_g.append(x_tmp.view(*x_tmp.shape[:2], -1))
        return x_g

    def split_efficient(self, x):
        x_g = []
        for i in range(len(self.groups_anc)):
            x_tmp = x[:, :, self.groups_anc[i], :]
            x_g.append(x_tmp)
        x_g = torch.cat(x_g, -2)
        return x_g


    def merge_efficient(self, x_b, b_i):
        ## x_b (b,f,4+4+6+4+4, 32)

        x_anchor = x_b[:, :, self.anchor_list, :]
        x_anchor = torch.mean(x_anchor, -2, keepdim=True)
        x_b = torch.cat([x_b, x_anchor], -2)
        x_b = x_b[:, :, self.other_list, :]
        return x_b

    def merge(self, x_b, b_i):
        b, f, _ = x_b[0].shape
        x_g = []
        anchor_0 = []
        anchor_8 = []

        for i in range(len(self.groups_anc)):
            p = self.groups_anc[i].shape[0] - 1
            if self.anchors[i] == 0:
                anchor_0.append(x_b[i].clone().view(b, f, p+1, -1)[:, :, 0:1, :])
            else:
                anchor_8.append(x_b[i].clone().view(b, f, p+1, -1)[:, :, 0:1, :])
        anchor_8.append(x_b[2].clone().view(b, f, 6, -1)[:, :, 3:4, :])
        anchor_8 = torch.stack(anchor_8, 0).mean(0)
        if self.merge_type == 'mean':
            anchor_0 = torch.stack(anchor_0, 0).mean(0)
        else:
            anchor_0 = torch.cat(anchor_0, 2).view(b*f, len(self.groups_anc), -1)
            anchor_0 = self.merge_blocks[b_i](anchor_0)
            anchor_0 = anchor_0.view(b, f, len(self.groups_anc), -1)

        for i in range(len(self.groups_anc)):
            p = self.groups_anc[i].shape[0] - 1
            if self.anchors[i] == 0:
                if anchor_0.shape[2] == 1:
                    x_tmp = torch.cat([anchor_0, x_b[i].clone().view(b, f, p + 1, -1)[:, :, 1:, :]], dim=2)
                else:
                    x_tmp = torch.cat([anchor_0[:, :, i:i+1, :], x_b[i].clone().view(b, f, p + 1, -1)[:, :, 1:, :]], dim=2)
            else:
                x_tmp = torch.cat([anchor_8, x_b[i].clone().view(b, f, p + 1, -1)[:, :, 1:, :]], dim=2)
            x_g.append(x_tmp.view(b, f, -1))
        return x_g




    def Spatial_forward_features(self, x):
        b, f, p, _ = x.shape  ##### b is batch size
        x = rearrange(x, 'b f p c -> (b f) p c')
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        for blk in self.Spatial_blocks:
            x = blk(x)
        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) p c -> b f (p c)', f=f)
        # if not self.ST:
        #     x = x.mean(1)
        return x


    def Temporal_efficient(self, x):
        b, _, f, p = x.shape
        x = rearrange(x, 'b c f p -> (b f) p c')
        x = self.patch_to_embedding(x)
        x = rearrange(x, '(b f) p c -> b f p c', f=f)
        x_g = self.split_efficient(x)
        for blk_i in range(len(self.blocks)):
            blk = self.blocks[blk_i]
            x_b = []
            for i in range(len(self.groups)):
                x_i = x_g[:, :, self.group_list[i]:self.group_list[i+1], :]
                x_i = x_i.view(b, x_i.shape[1], -1)
                x_i = blk[i](x_i)
                if blk_i == len(self.blocks) - 1:
                    x_i = self.Temporal_norm[i](x_i)
                    x_i = self.weighted_mean[i](x_i).view(b, 1, len(self.groups[i]) + 1, -1)
                    x_i = x_i[:, :, 1:, :]
                else:
                    x_i = x_i.view(b, x_i.shape[1], len(self.groups[i]) + 1, -1)
                x_b.append(x_i)
            x_b = torch.cat(x_b, -2)
            if self.Merge and blk_i < len(self.blocks) - 1:
                x_g = self.merge_efficient(x_b, blk_i)
            else:
                x_g = x_b.clone()

        x_out = x_g[:, :, self.inverse_group, :]
        # x_out = x_out.view(b, -1)
        # x_out = self.transform(x_out)
        x_out = x_out.view(b, 1, p, -1)

        return x_out


    def Temporal_forward_features(self, x):
        b, _, f, p = x.shape
        x = rearrange(x, 'b c f p -> (b f) p c')
        x = self.patch_to_embedding(x)
        x = rearrange(x, '(b f) p c -> b f p c', f=f)
        x_g = self.split(x)

        for blk_i in range(len(self.blocks)):
            blk = self.blocks[blk_i]
            x_b = []
            for i in range(len(self.groups)):
                x_i = x_g[i]
                x_i = blk[i](x_i)
                x_b.append(x_i)
            if self.Merge and blk_i < len(self.blocks):
                x_g = self.merge(x_b, blk_i)
            else:
                x_g = []
                for i in range(len(self.groups)):
                    x_g.append(x_b[i])
        x_out = []
        for i in range(len(self.groups)):
            x_i = x_g[i]
            x_i = self.Temporal_norm[i](x_i)  # (b, f', (p' c))
            x_i = self.weighted_mean[i](x_i).view(b, len(self.groups[i])+1, -1)
            x_i = x_i[:, 1:, :]
            x_out.append(x_i)

        x_out = torch.cat(x_out, 1)
        x_out = x_out[:, self.inverse_group, :]
        # x_out = x_out.view(b, -1)
        # x_out = self.transform(x_out)
        x_out = x_out.view(b, 1, p, -1)

        return x_out

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)

        b, _, _, p = x.shape

        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data
        x = self.Temporal_efficient(x)
        x = self.Spatial_forward_features(x)

        x = self.head(x)
        x = x.view(b, 1, p, -1)
        # x[:, 0, (11, 12, 13, 14, 15, 16), :] = x[:, 0, (11, 12, 13, 14, 15, 16), :] + x[:, 0, 8:9, :]
        return x



class DualShiftGroupTransformer(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None, merge=False, merge_type='mean', M=9, shift=3):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            merge (bool): merge anchors after each temporal block
            merge_type (str): 'mean' or 'transformer'
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints  #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = num_joints * 3  #### output dimension is num_joints * 3

        self.groups = ((1,2,3),(4,5,6),(0,7,8,9,10),(11,12,13),(14,15,16))
        self.anchors = (0,0,0,0,0)

        self.groups_anc = []
        if len(self.groups) > 0:
            self.inverse_group = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            count = 0
            for i in range(len(self.groups)):
                self.groups_anc.append(
                    torch.cat([torch.tensor([self.anchors[i]]).type(torch.long).cuda(non_blocking=True),
                               torch.tensor(self.groups[i]).type(torch.long).cuda(non_blocking=True)], dim=0))

                for g_idx in self.groups[i]:
                    self.inverse_group[g_idx] = count
                    count += 1
            self.inverse_group = torch.tensor(self.inverse_group).type(torch.long).cuda(non_blocking=True)

        self.anchor_list = torch.tensor((0, 4, 8, 14, 18), dtype=torch.long, device='cuda')
        self.other_list = torch.tensor((-1, 1, 2, 3, -1, 5, 6, 7, -1, 9, 10, 11, 12, 13, -1, 15, 16, 17, -1, 19, 20, 21),
                                  dtype=torch.long, device='cuda')

        self.group_list = [0, 4, 8, 14, 18, 22]


        ### spatial patch embedding
        self.patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)

        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))


        # self.Temporal_pos_embed = [self.Temporal_pos_embed_0, self.Temporal_pos_embed_1, self.Temporal_pos_embed_2, self.Temporal_pos_embed_3, self.Temporal_pos_embed_4]
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(2)])

        down_blocks = (np.log(num_frame // 9) / np.log(3)).astype(np.int32)

        self.blocks = nn.ModuleList([
            nn.ModuleList([
                ShiftBlock(
                    dim=embed_dim_ratio * (len(g) + 1), num_heads=num_heads,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    M=M, shift=shift, downsample=True)
                for g in self.groups
            ])
            for i in range(min(depth, down_blocks))])
        if down_blocks < depth:
            self.blocks.extend(nn.ModuleList([
                nn.ModuleList([
                    Block(
                        dim=embed_dim_ratio * (len(g) + 1), num_heads=num_heads,
                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                    for g in self.groups
                ])
                for i in range(down_blocks, depth)]))

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = nn.ModuleList([
            norm_layer(embed_dim_ratio * (len(g) + 1))
            for g in self.groups
        ])


        self.weighted_mean = nn.ModuleList([torch.nn.Conv1d(in_channels=9, out_channels=1, kernel_size=1)
                                            for g in self.groups])

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

        self.Merge = merge
        self.merge_type = merge_type
        if merge_type != 'mean':
            self.merge_blocks = nn.ModuleList([
                Block(
                    dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                for i in range(depth)
            ])
            # self.merge_norm = nn.ModuleList([
            #     norm_layer(embed_dim_ratio)
            #     for i in range(depth-1)
            # ])

    def split(self, x):
        x_g = []
        for i in range(len(self.groups_anc)):
            x_tmp = x[:, :, self.groups_anc[i], :]
            x_g.append(x_tmp.view(*x_tmp.shape[:2], -1))
        return x_g

    def split_efficient(self, x):
        x_g = []
        for i in range(len(self.groups_anc)):
            x_tmp = x[:, :, self.groups_anc[i], :]
            x_g.append(x_tmp)
        x_g = torch.cat(x_g, -2)
        return x_g


    def merge_efficient(self, x_b, b_i):
        ## x_b (b,f,4+4+6+4+4, 32)

        x_anchor = x_b[:, :, self.anchor_list, :]
        x_anchor = torch.mean(x_anchor, -2, keepdim=True)
        x_b = torch.cat([x_b, x_anchor], -2)
        x_b = x_b[:, :, self.other_list, :]
        return x_b

    def merge(self, x_b, b_i):
        b, f, _ = x_b[0].shape
        x_g = []
        anchor_0 = []
        anchor_8 = []

        for i in range(len(self.groups_anc)):
            p = self.groups_anc[i].shape[0] - 1
            if self.anchors[i] == 0:
                anchor_0.append(x_b[i].clone().view(b, f, p+1, -1)[:, :, 0:1, :])
            else:
                anchor_8.append(x_b[i].clone().view(b, f, p+1, -1)[:, :, 0:1, :])
        anchor_8.append(x_b[2].clone().view(b, f, 6, -1)[:, :, 3:4, :])
        anchor_8 = torch.stack(anchor_8, 0).mean(0)
        if self.merge_type == 'mean':
            anchor_0 = torch.stack(anchor_0, 0).mean(0)
        else:
            anchor_0 = torch.cat(anchor_0, 2).view(b*f, len(self.groups_anc), -1)
            anchor_0 = self.merge_blocks[b_i](anchor_0)
            anchor_0 = anchor_0.view(b, f, len(self.groups_anc), -1)

        for i in range(len(self.groups_anc)):
            p = self.groups_anc[i].shape[0] - 1
            if self.anchors[i] == 0:
                if anchor_0.shape[2] == 1:
                    x_tmp = torch.cat([anchor_0, x_b[i].clone().view(b, f, p + 1, -1)[:, :, 1:, :]], dim=2)
                else:
                    x_tmp = torch.cat([anchor_0[:, :, i:i+1, :], x_b[i].clone().view(b, f, p + 1, -1)[:, :, 1:, :]], dim=2)
            else:
                x_tmp = torch.cat([anchor_8, x_b[i].clone().view(b, f, p + 1, -1)[:, :, 1:, :]], dim=2)
            x_g.append(x_tmp.view(b, f, -1))
        return x_g




    def Spatial_forward_features(self, x):
        b, f, p, _ = x.shape  ##### b is batch size
        x = rearrange(x, 'b f p c -> (b f) p c')
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        for blk in self.Spatial_blocks:
            x = blk(x)
        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) p c -> b f (p c)', f=f)
        # if not self.ST:
        #     x = x.mean(1)
        return x


    def Temporal_efficient(self, x):
        b, _, f, p = x.shape
        x = rearrange(x, 'b c f p -> (b f) p c')
        x = self.patch_to_embedding(x)
        x = rearrange(x, '(b f) p c -> b f p c', f=f)
        x_g = self.split_efficient(x)
        for blk_i in range(len(self.blocks)):
            blk = self.blocks[blk_i]
            x_b = []
            for i in range(len(self.groups)):
                x_i = x_g[:, :, self.group_list[i]:self.group_list[i+1], :]
                x_i = x_i.view(b, x_i.shape[1], -1)
                x_i = blk[i](x_i)
                if blk_i == len(self.blocks) - 1:
                    x_i = self.Temporal_norm[i](x_i)
                    x_i = self.weighted_mean[i](x_i).view(b, 1, len(self.groups[i]) + 1, -1)
                    x_i = x_i[:, :, 1:, :]
                else:
                    x_i = x_i.view(b, x_i.shape[1], len(self.groups[i]) + 1, -1)
                x_b.append(x_i)
            x_b = torch.cat(x_b, -2)
            if self.Merge and blk_i < len(self.blocks) - 1:
                x_g = self.merge_efficient(x_b, blk_i)
            else:
                x_g = x_b.clone()

        x_out = x_g[:, :, self.inverse_group, :]
        # x_out = x_out.view(b, -1)
        # x_out = self.transform(x_out)
        x_out = x_out.view(b, 1, p, -1)

        return x_out


    def Temporal_forward_features(self, x):
        b, _, f, p = x.shape
        x = rearrange(x, 'b c f p -> (b f) p c')
        x = self.patch_to_embedding(x)
        x = rearrange(x, '(b f) p c -> b f p c', f=f)
        x_g = self.split(x)

        for blk_i in range(len(self.blocks)):
            blk = self.blocks[blk_i]
            x_b = []
            for i in range(len(self.groups)):
                x_i = x_g[i]
                x_i = blk[i](x_i)
                x_b.append(x_i)
            if self.Merge and blk_i < len(self.blocks):
                x_g = self.merge(x_b, blk_i)
            else:
                x_g = []
                for i in range(len(self.groups)):
                    x_g.append(x_b[i])
        x_out = []
        for i in range(len(self.groups)):
            x_i = x_g[i]
            x_i = self.Temporal_norm[i](x_i)  # (b, f', (p' c))
            x_i = self.weighted_mean[i](x_i).view(b, len(self.groups[i])+1, -1)
            x_i = x_i[:, 1:, :]
            x_out.append(x_i)

        x_out = torch.cat(x_out, 1)
        x_out = x_out[:, self.inverse_group, :]
        # x_out = x_out.view(b, -1)
        # x_out = self.transform(x_out)
        x_out = x_out.view(b, 1, p, -1)

        return x_out

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)

        b, _, _, p = x.shape

        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data
        x = self.Temporal_efficient(x)
        x = self.Spatial_forward_features(x)

        x = self.head(x)
        x = x.view(b, 1, p, -1)
        # x[:, 0, (11, 12, 13, 14, 15, 16), :] = x[:, 0, (11, 12, 13, 14, 15, 16), :] + x[:, 0, 8:9, :]
        return x



class Refine(nn.Module):

    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=2,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., norm_layer=None):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = num_joints * embed_dim_ratio
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0.0, norm_layer=norm_layer)
            for i in range(depth)])
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0.0, norm_layer=norm_layer)
            for i in range(depth)])
        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_joints*2),
        )

    def Spatial_forward_features(self, x):
        b, _, f, p = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        x = rearrange(x, 'b c f p  -> (b f) p  c', )

        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)

        for blk in self.Spatial_blocks:
            x = blk(x)

        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) w c -> b f (w c)', f=f)
        return x

    def forward_features(self, x):
        b, f = x.shape[0:2]
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)

        x = self.Temporal_norm(x)
        ##### x size [b, f, emb_dim], then take weighted mean on frame dimension, we only predict 3D pose of the center frame
        x = x.view(b, f, -1)
        return x

    def forward(self, x):
        b, _, f, p = x.shape
        x = self.Spatial_forward_features(x)
        x = self.forward_features(x)
        x = self.head(x)
        x = x.view(b, f, p, -1)
        x = rearrange(x, 'b f p c -> b c f p')
        return x


class PoseRefineTransformer(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None, down_factor=1, merge=False, merge_type='mean', ST=False, pos_type='learnable', weighted=False):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            merge (bool): merge anchors after each temporal block
            merge_type (str): 'mean' or 'transformer'
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints  #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = num_joints * 3  #### output dimension is num_joints * 3

        self.groups = ((1,2,3),(4,5,6),(0,7,8,9,10),(11,12,13),(14,15,16))
        self.anchors = (0,0,0,0,0)

        self.groups_anc = []
        if len(self.groups) > 0:
            self.inverse_group = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            count = 0
            for i in range(len(self.groups)):
                self.groups_anc.append(
                    torch.cat([torch.tensor([self.anchors[i]]).type(torch.long).cuda(non_blocking=True),
                               torch.tensor(self.groups[i]).type(torch.long).cuda(non_blocking=True)], dim=0))

                for g_idx in self.groups[i]:
                    self.inverse_group[g_idx] = count
                    count += 1
            self.inverse_group = torch.tensor(self.inverse_group).type(torch.long).cuda(non_blocking=True)

        self.refine = Refine(num_frame=num_frame, num_joints=num_joints, in_chans=in_chans, embed_dim_ratio=embed_dim_ratio//2, depth=2,
                 num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, norm_layer=norm_layer)

        ### spatial patch embedding
        self.patch_to_embedding = nn.Linear(in_chans * 2, embed_dim_ratio)

        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))

        if pos_type == 'learnable':
            self.Temporal_pos_embed_0 = nn.Parameter(torch.zeros(1, num_frame, embed_dim_ratio * (len(self.groups[0]) + 1)))
            self.Temporal_pos_embed_1 = nn.Parameter(torch.zeros(1, num_frame, embed_dim_ratio * (len(self.groups[1]) + 1)))
            self.Temporal_pos_embed_2 = nn.Parameter(torch.zeros(1, num_frame, embed_dim_ratio * (len(self.groups[2]) + 1)))
            self.Temporal_pos_embed_3 = nn.Parameter(torch.zeros(1, num_frame, embed_dim_ratio * (len(self.groups[3]) + 1)))
            self.Temporal_pos_embed_4 = nn.Parameter(torch.zeros(1, num_frame, embed_dim_ratio * (len(self.groups[4]) + 1)))
        else:
            self.Temporal_pos_embed_0 = nn.Parameter(self.get_pos_embed(num_frame, embed_dim_ratio * (len(self.groups[0]) + 1)), requires_grad=False)
            self.Temporal_pos_embed_1 = nn.Parameter(self.get_pos_embed(num_frame, embed_dim_ratio * (len(self.groups[1]) + 1)), requires_grad=False)
            self.Temporal_pos_embed_2 = nn.Parameter(self.get_pos_embed(num_frame, embed_dim_ratio * (len(self.groups[2]) + 1)), requires_grad=False)
            self.Temporal_pos_embed_3 = nn.Parameter(self.get_pos_embed(num_frame, embed_dim_ratio * (len(self.groups[3]) + 1)), requires_grad=False)
            self.Temporal_pos_embed_4 = nn.Parameter(self.get_pos_embed(num_frame, embed_dim_ratio * (len(self.groups[4]) + 1)), requires_grad=False)


        # self.Temporal_pos_embed = [self.Temporal_pos_embed_0, self.Temporal_pos_embed_1, self.Temporal_pos_embed_2, self.Temporal_pos_embed_3, self.Temporal_pos_embed_4]
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        down_blocks = (np.log(num_frame // 9) / np.log(3)).astype(np.int32)

        self.blocks = nn.ModuleList([
            nn.ModuleList([
                TBlock(
                    dim=embed_dim_ratio * (len(g) + 1), num_heads=num_heads,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    down_factor=down_factor)
                for g in self.groups
            ])
            for i in range(min(depth, down_blocks))])
        if down_blocks < depth:
            self.blocks.extend(nn.ModuleList([
                nn.ModuleList([
                    TBlock(
                        dim=embed_dim_ratio * (len(g) + 1), num_heads=num_heads,
                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                        down_factor=1)
                    for g in self.groups
                ])
                for i in range(down_blocks, depth)]))

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = nn.ModuleList([
            norm_layer(embed_dim_ratio * (len(g) + 1))
            for g in self.groups
        ])

        # self.transform = nn.Linear(embed_dim, embed_dim)

        ####### A easy way to implement weighted mean
        # if attention_type == 'Down':
        #     self.weighted_mean = torch.nn.Conv1d(in_channels=num_frame // (factor_q ** depth), out_channels=1, kernel_size=1)
        # else:
        if weighted:
            self.weighted_mean = nn.ModuleList([torch.nn.Conv1d(in_channels=num_frame, out_channels=1, kernel_size=1)
                                                for g in self.groups])
        else:
            self.weighted_mean = nn.ModuleList([MEAN() for g in self.groups])

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim_ratio),
            nn.Linear(embed_dim_ratio, 3),
        )
        self.int_head = nn.Sequential(
            nn.LayerNorm(embed_dim_ratio),
            nn.Linear(embed_dim_ratio, 3),
        )

        self.Merge = merge
        self.merge_type = merge_type
        if merge_type != 'mean':
            self.merge_blocks = nn.ModuleList([
                Block(
                    dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                for i in range(depth)
            ])
            # self.merge_norm = nn.ModuleList([
            #     norm_layer(embed_dim_ratio)
            #     for i in range(depth-1)
            # ])
        self.ST = ST

    def get_pos_embed(self, seq, dim):
        pad = seq // 2
        pos = torch.arange(-pad, pad+1, dtype=torch.float32)
        pos = pos.view(-1, 1).repeat(1, dim//2)
        div = torch.arange(0, dim//2, dtype=torch.float32)
        div = div.view(1, -1).repeat(seq, 1)
        div = 10000 ** (2.0 * div / dim)
        PE_sin = torch.sin(pos / div)
        PE_cos = torch.cos(pos / div)
        PE = torch.stack([PE_sin, PE_cos], 2).view(1, seq, dim)
        return PE



    def split(self, x):
        x_g = []
        for i in range(len(self.groups_anc)):
            x_tmp = x[:, :, self.groups_anc[i], :]
            x_g.append(x_tmp.view(*x_tmp.shape[:2], -1))
        return x_g

    def merge(self, x_b, b_i):
        b, f, _ = x_b[0].shape
        x_g = []
        anchor_0 = []
        anchor_8 = []

        for i in range(len(self.groups_anc)):
            p = self.groups_anc[i].shape[0] - 1
            if self.anchors[i] == 0:
                anchor_0.append(x_b[i].clone().view(b, f, p+1, -1)[:, :, 0:1, :])
            else:
                anchor_8.append(x_b[i].clone().view(b, f, p+1, -1)[:, :, 0:1, :])
        anchor_8.append(x_b[2].clone().view(b, f, 6, -1)[:, :, 3:4, :])
        anchor_8 = torch.stack(anchor_8, 0).mean(0)
        if self.merge_type == 'mean':
            anchor_0 = torch.stack(anchor_0, 0).mean(0)
        else:
            anchor_0 = torch.cat(anchor_0, 2).view(b*f, len(self.groups_anc), -1)
            anchor_0 = self.merge_blocks[b_i](anchor_0)
            anchor_0 = anchor_0.view(b, f, len(self.groups_anc), -1)

        for i in range(len(self.groups_anc)):
            p = self.groups_anc[i].shape[0] - 1
            if self.anchors[i] == 0:
                if anchor_0.shape[2] == 1:
                    x_tmp = torch.cat([anchor_0, x_b[i].clone().view(b, f, p + 1, -1)[:, :, 1:, :]], dim=2)
                else:
                    x_tmp = torch.cat([anchor_0[:, :, i:i+1, :], x_b[i].clone().view(b, f, p + 1, -1)[:, :, 1:, :]], dim=2)
            else:
                x_tmp = torch.cat([anchor_8, x_b[i].clone().view(b, f, p + 1, -1)[:, :, 1:, :]], dim=2)
            x_g.append(x_tmp.view(b, f, -1))
        return x_g




    def Spatial_forward_features(self, x):
        b, f, p, _ = x.shape  ##### b is batch size
        x = rearrange(x, 'b f p c -> (b f) p c')
        if self.ST:
            x = self.patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        for blk in self.Spatial_blocks:
            x = blk(x)
        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) p c -> b f p c', f=f)
        # if not self.ST:
        #     x = x.mean(1)
        return x

    def Temporal_forward_features(self, x):
        if self.ST:
            x = rearrange(x, 'b f p c -> b c f p')
        b, _, f, p = x.shape
        x = rearrange(x, 'b c f p -> (b f) p c')
        if not self.ST:
            x = self.patch_to_embedding(x)
        x = rearrange(x, '(b f) p c -> b f p c', f=f)
        x_g = self.split(x)
        x_b = []
        # for i in range(len(self.groups)):
        #     x_i = x_g[i]
        #     x_i += self.Temporal_pos_embed[i]
        #     x_i = self.pos_drop(x_i)
        #     x_b.append(x_i)

        x_0 = x_g[0]
        x_0 += self.Temporal_pos_embed_0
        x_0 = self.pos_drop(x_0)
        x_b.append(x_0)

        x_1 = x_g[1]
        x_1 += self.Temporal_pos_embed_1
        x_1 = self.pos_drop(x_1)
        x_b.append(x_1)

        x_2 = x_g[2]
        x_2 += self.Temporal_pos_embed_2
        x_2 = self.pos_drop(x_2)
        x_b.append(x_2)

        x_3 = x_g[3]
        x_3 += self.Temporal_pos_embed_3
        x_3 = self.pos_drop(x_3)
        x_b.append(x_3)

        x_4 = x_g[4]
        x_4 += self.Temporal_pos_embed_4
        x_4 = self.pos_drop(x_4)
        x_b.append(x_4)


        x_g = []
        for i in range(len(self.groups)):
            x_g.append(x_b[i])
        for blk_i in range(len(self.blocks)):
            blk = self.blocks[blk_i]
            x_b = []
            for i in range(len(self.groups)):
                x_i = x_g[i]
                x_i = blk[i](x_i)
                x_b.append(x_i)
            if self.Merge and blk_i < len(self.blocks):
                x_g = self.merge(x_b, blk_i)
            else:
                x_g = []
                for i in range(len(self.groups)):
                    x_g.append(x_b[i])
        x_out = []
        for i in range(len(self.groups)):
            x_i = x_g[i]
            x_i = self.Temporal_norm[i](x_i)  # (b, f', (p' c))
            x_i = self.weighted_mean[i](x_i).view(b, len(self.groups[i])+1, -1)
            # x_i = x_i.mean(1).view(b, len(self.groups[i])+1, -1)
            x_i = x_i[:, 1:, :]
            x_out.append(x_i)

        x_out = torch.cat(x_out, 1)
        x_out = x_out[:, self.inverse_group, :]
        # x_out = x_out.view(b, -1)
        # x_out = self.transform(x_out)
        x_out = x_out.view(b, 1, p, -1)

        return x_out

    def forward(self, x, int=False, refine=False):
        x = x.permute(0, 3, 1, 2)

        b, _, _, p = x.shape

        x_refine = self.refine(x)
        x = torch.cat([x, x_refine], 1)

        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data
        if self.ST:
            x = rearrange(x, 'b c f p -> b f p c')
            x = self.Spatial_forward_features(x)
            x = self.Temporal_forward_features(x)
        else:
            x = self.Temporal_forward_features(x)
            x = self.Spatial_forward_features(x)

        x = self.head(x)
        x = x.view(b, 1, p, -1)
        x_refine = rearrange(x_refine, 'b c f p -> b f p c')
        # x[:, 0, (11, 12, 13, 14, 15, 16), :] = x[:, 0, (11, 12, 13, 14, 15, 16), :] + x[:, 0, 8:9, :]
        if refine:
            return x, x_refine
        return x


class PoseGroupShareTransformer(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None, down_factor=1, merge=False):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * 4  #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = num_joints * 3  #### output dimension is num_joints * 3

        self.groups = ((1,2,3),(4,5,6),(11,12,13),(14,15,16),(7,8,9,10))
        self.anchors = (0,0,0,0,0)

        self.groups_anc = []
        if len(self.groups) > 0:
            self.inverse_group = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # total_length=16, omitting the root node
            count = 0
            for i in range(len(self.groups)):
                self.groups_anc.append(
                    torch.cat([torch.tensor([self.anchors[i]]).type(torch.long).cuda(non_blocking=True),
                               torch.tensor(self.groups[i]).type(torch.long).cuda(non_blocking=True)], dim=0))

                for g_idx in self.groups[i]:
                    self.inverse_group[g_idx-1] = count
                    count += 1
            self.inverse_group = torch.tensor(self.inverse_group).type(torch.long).cuda()


        ### spatial patch embedding
        self.patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)

        self.init_embedding_limbs = nn.Linear(embed_dim_ratio * 4, embed_dim)
        self.init_embedding_torso = nn.Linear(embed_dim_ratio * 5, embed_dim)

        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints - 1, embed_dim_ratio))

        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        down_blocks = (np.log(num_frame // 9) / np.log(3)).astype(np.int32)

        self.J2E_limbs = nn.Linear(embed_dim_ratio * 4, embed_dim)
        self.E2J_limbs = nn.Linear(embed_dim, embed_dim_ratio * 4)

        self.J2E_torso = nn.Linear(embed_dim_ratio * 5, embed_dim)
        self.E2J_torso = nn.Linear(embed_dim, embed_dim_ratio * 5)


        self.blocks = nn.ModuleList([
                TBlock(
                    dim=embed_dim, num_heads=num_heads,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    down_factor=down_factor)
                for i in range(min(depth, down_blocks))])
        if down_blocks < depth:
            self.blocks.extend(nn.ModuleList([
                    TBlock(
                        dim=embed_dim, num_heads=num_heads,
                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                        down_factor=1)
                    for i in range(down_blocks, depth)]))

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)


        # self.transform = nn.Linear(embed_dim, embed_dim)

        ####### A easy way to implement weighted mean
        # if attention_type == 'Down':
        #     self.weighted_mean = torch.nn.Conv1d(in_channels=num_frame // (factor_q ** depth), out_channels=1, kernel_size=1)
        # else:
        #     self.weighted_mean = torch.nn.Conv1d(in_channels=num_frame, out_channels=1, kernel_size=1)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim_ratio),
            nn.Linear(embed_dim_ratio, 3),
        )

        self.Merge = merge

        if self.Merge:
            self.merge_blocks = nn.ModuleList([
                Block(
                    dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                for i in range(depth)
            ])

    def split(self, x):
        x_g = []
        for i in range(len(self.groups_anc)):
            x_tmp = x[:, :, self.groups_anc[i], :]
            x_tmp = x_tmp.view(*x_tmp.shape[:2], -1)  # (batch, frames, 32* 4 or 5)
            if i != len(self.groups_anc) - 1:
                x_tmp = self.init_embedding_limbs(x_tmp)
            else:
                x_tmp = self.init_embedding_torso(x_tmp)
            x_g.append(x_tmp)
        x_g = torch.cat(x_g, 0)
        return x_g

    def merge(self, x_g, i):
        B, f, _ = x_g.shape
        b = B // 5
        x_g = x_g.view(5, b, f, -1)
        x_limbs = self.E2J_limbs(x_g[:4, :, :, :].permute(1,2,0,3))  # (b, f, 4, 4*embed_ratio)
        x_torso = self.E2J_torso(x_g[4:,:, :, :].permute(1,2,0,3))  # (b, f, 1, 5*embed_ratio)
        mut_limbs = x_limbs.view(b,f,4,4,-1)[:, :, :, 0, :]
        mut_torso = x_torso.view(b,f,1,5,-1)[:, :, :, 0, :]
        embed_ratio = mut_limbs.shape[-1]
        x_mut = torch.cat([mut_limbs.view(b*f, 4, -1), mut_torso.view(b*f, 1, -1)], 1)
        x_mut = self.merge_blocks[i](x_mut)
        x_mut = x_mut.view(b,f,5,-1)
        x_limbs = torch.cat([x_mut[:,:,:4,:], x_limbs[:, :, :, embed_ratio:]], -1)  # (b, f, 4, 4*embed_ratio)
        x_torso = torch.cat([x_mut[:, :, 4:, :], x_torso[:, :, :, embed_ratio:]], -1)  # (b, f, 1, 5*embed_ratio)
        x_limbs = self.J2E_limbs(x_limbs).permute(2, 0, 1, 3)
        x_torso = self.J2E_torso(x_torso).permute(2, 0, 1, 3)
        x_g = torch.cat([x_limbs, x_torso], 0)
        x_g = x_g.view(B, f, -1)
        return x_g




    def Spatial_forward_features(self, x):
        b, f, p, _ = x.shape  ##### b is batch size
        x = rearrange(x, 'b f p c -> (b f) p c')
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        for blk in self.Spatial_blocks:
            x = blk(x)
        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) p c -> b f p c', f=f)
        x = x.mean(1)
        return x

    def Temporal_forward_features(self, x):
        b, _, f, p = x.shape
        x = rearrange(x, 'b c f p -> (b f) p c')
        x = self.patch_to_embedding(x)
        x = rearrange(x, '(b f) p c -> b f p c', f=f)
        x_g = self.split(x)  # (batch * 5, frames, embed)
        x_g += self.Temporal_pos_embed
        x_g = self.pos_drop(x_g)
        for i in range(len(self.blocks)):
            blk = self.blocks[i]
            x_g = blk(x_g)  # (batch * 5, frames, embed)
            if self.Merge:
                x_g = self.merge(x_g, i)
        x_g = self.Temporal_norm(x_g)  # (b * 5, f', embed)
        x_g = x_g.mean(1).view(5, b, -1).permute(1, 0, 2).contiguous()
        x_limbs = self.E2J_limbs(x_g[:, (0,1,3,4), :]).view(b, 4, 4, -1)
        x_torso = self.E2J_torso(x_g[:, 2:3, :]).view(b, 1, 5, -1)
        # x_root = torch.cat([x_limbs[:, :, 0, :], x_torso[:, :, 0, :]], 1).mean(1, keepdim=True)  # (b, 1, embed_ratio)
        x_others = torch.cat([x_limbs[:, :, 1:, :].contiguous().view(b, 12, -1), x_torso[:, :, 1:, :].contiguous().view(b, 4, -1)], 1)
        x_others = x_others[:, self.inverse_group, :]
        x_out = x_others.view(b, 1, p-1, -1)
        return x_out

    def forward(self, x, int=False):
        x = x.permute(0, 3, 1, 2)

        b, _, _, p = x.shape

        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data
        x = self.Temporal_forward_features(x)
        x = self.Spatial_forward_features(x)
        x = self.head(x)
        x = x.view(b, 1, p-1, -1)
        x = torch.cat([torch.zeros(b, 1, 1, 3, device=x.device).float(), x], 2)
        # x[:, 0, (11, 12, 13, 14, 15, 16), :] = x[:, 0, (11, 12, 13, 14, 15, 16), :] + x[:, 0, 8:9, :]
        return x


class PosePyramidTransformer(nn.Module):

    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None, down_factor=1):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            merge (bool): merge anchors after each temporal block
            merge_type (str): 'mean' or 'transformer'
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints  #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = num_joints * 3  #### output dimension is num_joints * 3

        self.groups = ((1,2,3),(4,5,6),(0,7,8,9,10),(11,12,13),(14,15,16))
        self.anchors = (0,0,0,0,0)

        self.groups_anc = []
        if len(self.groups) > 0:
            self.inverse_group = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            count = 0
            for i in range(len(self.groups)):
                self.groups_anc.append(
                    torch.cat([torch.tensor([self.anchors[i]]).type(torch.long).cuda(non_blocking=True),
                               torch.tensor(self.groups[i]).type(torch.long).cuda(non_blocking=True)], dim=0))

                for g_idx in self.groups[i]:
                    self.inverse_group[g_idx] = count
                    count += 1
            self.inverse_group = torch.tensor(self.inverse_group).type(torch.long).cuda(non_blocking=True)


        ### spatial patch embedding
        self.patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)

        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))


class ParallelSpatialTemporalTransformer(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=256, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim_spatial = embed_dim_ratio * num_frame  #### temporal embed_dim is the same as the spatial embed_dim
        embed_dim_temporal = embed_dim_ratio * 17
        out_dim = 3  #### output dimension is num_joints * 3

        ### spatial patch embedding
        # self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        # self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        #
        # self.Temporal_patch_to_embedding = nn.Linear(in_chans, embed_dim)
        # self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))

        self.patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frame, num_joints, embed_dim_ratio))
        #
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_spatial, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.Temporal_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_temporal, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.transforms = nn.ModuleList([
            Mlp(in_features=embed_dim_ratio, hidden_features=embed_dim_ratio, out_features=embed_dim_ratio)
            for _ in range(depth)
        ])

        self.norm = norm_layer(embed_dim_ratio)
        self.final_norm = norm_layer(embed_dim_ratio)
        # self.Temporal_norm = norm_layer(embed_dim)

        ####### A easy way to implement weighted mean
        self.weighted_mean = torch.nn.Conv2d(in_channels=num_frame, out_channels=1, kernel_size=1)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim_ratio),
            nn.Linear(embed_dim_ratio, out_dim),
        )

    def forward_features(self, x):
        b, _, f, p = x.shape
        # spatial_x = x.clone()
        # temporal_x = x.clone()
        # spatial_x = rearrange(spatial_x, 'b c f p  -> (b f) p c', )
        # spatial_x = self.Spatial_patch_to_embedding(spatial_x)
        # spatial_x += self.Spatial_pos_embed
        # spatial_x = self.pos_drop(spatial_x)
        # temporal_x = rearrange(temporal_x, 'b c f p -> (b p) f c', )
        # temporal_x = self.Temporal_patch_to_embedding(temporal_x)
        # temporal_x += self.Temporal_pos_embed
        # temporal_x = self.pos_drop(temporal_x)

        # spatial_x = rearrange(spatial_x, '(b f) p c -> b f p c', f=f)
        # temporal_x = rearrange(temporal_x, '(b p) f c -> b f p c', p=p)

        # x = temporal_x + spatial_x
        x = rearrange(x, 'b c f p -> b f p c')
        x = self.patch_to_embedding(x)
        x += self.pos_embed
        x = self.pos_drop(x)

        for i in range(len(self.Spatial_blocks)):
            spatial_x = rearrange(x, 'b f p c -> b p (f c)', )
            temporal_x = rearrange(x, 'b f p c -> b f (p c)', )
            spatial_x = self.Spatial_blocks[i](spatial_x)
            temporal_x = self.Temporal_blocks[i](temporal_x)
            spatial_x = rearrange(spatial_x, 'b p (f c) -> b f p c', f=f)
            temporal_x = rearrange(temporal_x, 'b f (p c) -> b f p c', p=p)
            # x = torch.cat([spatial_x, temporal_x], dim=-1)
            x = spatial_x + temporal_x
            x = x + self.transforms[i](self.norm(x))
            # x = self.norm(x)
            # x = self.transforms[i](x.view(b*f*p, -1)).view(b, f, p, -1)
            # if i != len(self.Spatial_blocks) - 1:
            #     spatial_x = rearrange(x, 'b f p c -> (b f) p c', )
            #     temporal_x = rearrange(x, 'b f p c -> (b p) f c', )
        x = self.final_norm(x)
        x = self.weighted_mean(x)
        x = x.view(b * p, -1)
        return x

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        b, _, _, p = x.shape
        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data
        x = self.forward_features(x)
        x = self.head(x)
        x = x.view(b, 1, p, -1)
        return x


class AlternatingSpatialTemporalTransformer(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=256, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim_spatial = embed_dim_ratio * num_frame  #### temporal embed_dim is the same as the spatial embed_dim
        embed_dim_temporal = embed_dim_ratio * 17
        out_dim = 3  #### output dimension is num_joints * 3

        ### spatial patch embedding
        # self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim)
        # self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim))
        #
        # self.Temporal_patch_to_embedding = nn.Linear(in_chans, embed_dim)
        # self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))

        self.patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frame, num_joints, embed_dim_ratio))
        #
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_spatial, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.Temporal_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_temporal, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        # self.transforms = nn.ModuleList([
        #     Mlp(in_features=embed_dim, hidden_features=embed_dim, out_features=embed_dim)
        #     for _ in range(depth)
        # ])

        # self.norm = norm_layer(embed_dim)
        self.final_norm = norm_layer(embed_dim_ratio)
        # self.Temporal_norm = norm_layer(embed_dim)

        ####### A easy way to implement weighted mean
        self.weighted_mean = torch.nn.Conv2d(in_channels=num_frame, out_channels=1, kernel_size=1)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim_ratio),
            nn.Linear(embed_dim_ratio, out_dim),
        )

    def forward_features(self, x):
        b, _, f, p = x.shape
        # spatial_x = x.clone()
        # temporal_x = x.clone()
        # spatial_x = rearrange(spatial_x, 'b c f p  -> (b f) p c', )
        # spatial_x = self.Spatial_patch_to_embedding(spatial_x)
        # spatial_x += self.Spatial_pos_embed
        # spatial_x = self.pos_drop(spatial_x)
        # temporal_x = rearrange(temporal_x, 'b c f p -> (b p) f c', )
        # temporal_x = self.Temporal_patch_to_embedding(temporal_x)
        # temporal_x += self.Temporal_pos_embed
        # temporal_x = self.pos_drop(temporal_x)

        # spatial_x = rearrange(spatial_x, '(b f) p c -> b f p c', f=f)
        # temporal_x = rearrange(temporal_x, '(b p) f c -> b f p c', p=p)

        # x = temporal_x + spatial_x
        x = rearrange(x, 'b c f p -> b f p c', )
        x = self.patch_to_embedding(x)
        x += self.pos_embed
        x = self.pos_drop(x)

        for i in range(len(self.Spatial_blocks)):
            x = rearrange(x, 'b f p c -> b p (f c)', )
            x = self.Spatial_blocks[i](x)
            x = rearrange(x, 'b p (f c) -> b f (p c)', f=f)
            x = self.Temporal_blocks[i](x)
            x = rearrange(x, 'b f (p c) -> b f p c', p=p)
            # x = self.norm(x)
            # x = self.transforms[i](x.view(b*f*p, -1)).view(b, f, p, -1)
            # if i != len(self.Spatial_blocks) - 1:
            #     spatial_x = rearrange(x, 'b f p c -> (b f) p c', )
            #     temporal_x = rearrange(x, 'b f p c -> (b p) f c', )
        x = self.final_norm(x)
        x = self.weighted_mean(x)
        x = x.view(b * p, -1)
        return x

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        b, _, _, p = x.shape
        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data
        x = self.forward_features(x)
        x = self.head(x)
        x = x.view(b, 1, p, -1)
        return x


class BalancedParallelTransformer(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=256, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_frame  #### temporal embed_dim is the same as the spatial embed_dim
        embed_dim_temporal = embed_dim_ratio * 17
        out_dim = 3  #### output dimension is num_joints * 3

        ### spatial patch embedding
        # self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        # self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        #
        # self.Temporal_patch_to_embedding = nn.Linear(in_chans, embed_dim)
        # self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))

        self.patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frame, num_joints, embed_dim_ratio))
        #
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.Temporal_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim_temporal, embed_dim),
                Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                      drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer),
                nn.Linear(embed_dim, embed_dim_temporal),
            )
            for i in range(depth)])

        self.transforms = nn.ModuleList([
            Mlp(in_features=embed_dim_ratio, hidden_features=embed_dim_ratio, out_features=embed_dim_ratio)
            for _ in range(depth)
        ])

        self.norm = nn.ModuleList([
            norm_layer(embed_dim_ratio)
            for _ in range(depth)
        ])
        self.final_norm = norm_layer(embed_dim_ratio)
        # self.Temporal_norm = norm_layer(embed_dim)

        ####### A easy way to implement weighted mean
        self.weighted_mean = torch.nn.Conv2d(in_channels=num_frame, out_channels=1, kernel_size=1)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim_ratio),
            nn.Linear(embed_dim_ratio, out_dim),
        )

    def forward_features(self, x):
        b, _, f, p = x.shape
        # spatial_x = x.clone()
        # temporal_x = x.clone()
        # spatial_x = rearrange(spatial_x, 'b c f p  -> (b f) p c', )
        # spatial_x = self.Spatial_patch_to_embedding(spatial_x)
        # spatial_x += self.Spatial_pos_embed
        # spatial_x = self.pos_drop(spatial_x)
        # temporal_x = rearrange(temporal_x, 'b c f p -> (b p) f c', )
        # temporal_x = self.Temporal_patch_to_embedding(temporal_x)
        # temporal_x += self.Temporal_pos_embed
        # temporal_x = self.pos_drop(temporal_x)

        # spatial_x = rearrange(spatial_x, '(b f) p c -> b f p c', f=f)
        # temporal_x = rearrange(temporal_x, '(b p) f c -> b f p c', p=p)

        # x = temporal_x + spatial_x
        x = rearrange(x, 'b c f p -> b f p c')
        x = self.patch_to_embedding(x)
        x += self.pos_embed
        x = self.pos_drop(x)

        for i in range(len(self.Spatial_blocks)):
            spatial_x = rearrange(x, 'b f p c -> b p (f c)', )
            temporal_x = rearrange(x, 'b f p c -> b f (p c)', )
            spatial_x = self.Spatial_blocks[i](spatial_x)
            temporal_x = self.Temporal_blocks[i](temporal_x)
            spatial_x = rearrange(spatial_x, 'b p (f c) -> b f p c', f=f)
            temporal_x = rearrange(temporal_x, 'b f (p c) -> b f p c', p=p)
            # x = torch.cat([spatial_x, temporal_x], dim=-1)
            x = spatial_x + temporal_x
            x = x + self.transforms[i](self.norm[i](x))
            # x = self.norm(x)
            # x = self.transforms[i](x.view(b*f*p, -1)).view(b, f, p, -1)
            # if i != len(self.Spatial_blocks) - 1:
            #     spatial_x = rearrange(x, 'b f p c -> (b f) p c', )
            #     temporal_x = rearrange(x, 'b f p c -> (b p) f c', )
        x = self.final_norm(x)
        x = self.weighted_mean(x)
        x = x.view(b * p, -1)
        return x

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        b, _, _, p = x.shape
        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data
        x = self.forward_features(x)
        x = self.head(x)
        x = x.view(b, 1, p, -1)
        return x


