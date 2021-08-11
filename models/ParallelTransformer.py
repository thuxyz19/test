import torch
import torch.nn as nn
from einops import rearrange
from functools import partial
from timm.models.layers import DropPath


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
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
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
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MultiHeadTransformer(nn.Module):
    """
    The multi-head transformer architecture to combine information across views, temporal sequence and spatial connections.
    The input features are locally pooled features of key joints together with their 2D coordinates.

    """
    def __init__(self, in_channels, embed=512, num_heads=8, n_views=4, seq_len=5, num_joints=17, depth=4, drop_path_rate=0.0, norm_layer=None, mlp_ratio=2.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, use_confidences=True):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.in_channels = in_channels
        self.n_views = n_views
        self.seq_len = seq_len
        self.num_joints = num_joints
        self.depth = depth
        self.embed = embed
        view_embed = self.in_channels * self.seq_len * self.num_joints
        spatial_embed = self.in_channels * self.n_views * self.seq_len
        temporal_embed = self.in_channels * self.n_views * self.num_joints

        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_views, self.seq_len, self.num_joints, self.in_channels))
        self.drop_pos = nn.Dropout(p=drop)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.View_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(view_embed, self.embed),
                Block(dim=self.embed, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                      drop=drop, attn_drop=attn_drop, drop_path=dpr[i], norm_layer=norm_layer),
                nn.Linear(self.embed, view_embed),
            )
            for i in range(self.depth)
        ])
        self.Spatial_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(spatial_embed, self.embed),
                Block(dim=self.embed, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                      drop=drop, attn_drop=attn_drop, drop_path=dpr[i], norm_layer=norm_layer),
                nn.Linear(self.embed, spatial_embed),
            )
            for i in range(self.depth)
        ])
        self.Temporal_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(temporal_embed, self.embed),
                Block(dim=self.embed, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                      drop=drop, attn_drop=attn_drop, drop_path=dpr[i], norm_layer=norm_layer),
                nn.Linear(self.embed, temporal_embed),
            )
            for i in range(self.depth)
        ])
        self.transforms = nn.ModuleList([
            Mlp(in_features=self.in_channels, hidden_features=self.in_channels, out_features=self.in_channels)
            for _ in range(self.depth)
        ])
        self.norm = nn.ModuleList([
            norm_layer(self.in_channels)
            for _ in range(self.depth)
        ])
        self.final_norm = norm_layer(self.in_channels)
        if use_confidences:
            self.head = Mlp(in_features=self.in_channels, hidden_features=self.in_channels, out_features=3)
        else:
            self.head = Mlp(in_features=self.in_channels, hidden_features=self.in_channels, out_features=2)

    def forward(self, x):
        """
        param x: B, C, Views, T, J
        output y: B, Views, T, J, 3
        """
        B, C, Views, T, J = x.shape
        x = rearrange(x, 'b c v t j -> b v t j c', )
        x = x + self.pos_embed
        x = self.drop_pos(x)

        for i in range(self.depth):
            view_x = rearrange(x, 'b v t j c -> b v (t j c)', )
            spatial_x = rearrange(x, 'b v t j c -> b j (v t c)', )
            temporal_x = rearrange(x, 'b v t j c -> b t (v j c)', )
            view_x = self.View_blocks[i](view_x)
            spatial_x = self.Spatial_blocks[i](spatial_x)
            temporal_x = self.Temporal_blocks[i](temporal_x)
            view_x = rearrange(view_x, 'b v (t j c) -> b v t j c', t=T, j=J)
            spatial_x = rearrange(spatial_x, 'b j (v t c) -> b v t j c', v=Views, t=T)
            temporal_x = rearrange(temporal_x, 'b t (v j c) -> b v t j c', v=Views, j=J)
            x = view_x + spatial_x + temporal_x
            x = x + self.transforms[i](self.norm[i](x))

        x = self.final_norm(x)
        x = self.head(x)  # b v t j c -> b v t j 3 (2D coordinates + confidence)
        return x


class Transformer(nn.Module):
    def __init__(self, in_channels, use_confidences=True):
        super().__init__()
        self.in_channels = in_channels
        self.use_confidences = use_confidences
        self.encoder = MultiHeadTransformer(in_channels=self.in_channels, embed=512, n_views=4, seq_len=5, num_joints=17, depth=4, use_confidences=self.use_confidences)

    def forward(self, x):
        """
        N, Views, T, V, C = x.size()
        """
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.encoder(x)  # N, Views, T, V, 3
        predict_2d = x[:, :, :, :, :2]
        if self.use_confidences:
            alg_confidences = x[:, :, :, :, 2]
        else:
            alg_confidences = None
        return predict_2d, alg_confidences
