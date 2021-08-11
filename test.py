from common.model_poseformer import Block, Attention, \
    ProbSparseAttention, TBlock, PoseTransformerInverse, \
    PoseTransformer, PoseGroupTransformer, PoseGroupShareTransformer, SwinTransformer, \
    ShiftTransformer, MutualTransformer
from common.model_group import ShiftGroupTransformer, ExtTransformer
from common.model_ablation import NoExt, NoGroup

import torch
import torch.nn as nn
from functools import partial
import time

norm_layer = partial(nn.LayerNorm, eps=1e-6)


# b = Block(dim=32*17, num_heads=8, mlp_ratio=2, qkv_bias=True, qk_scale=None,
#           drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=norm_layer, attn_type=ProbSparseAttention, factor_k=5, factor_q=1).cuda()
# a = Attention(32*17, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0).cuda()
# p = ProbSparseAttention(32*17, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0, factor_k=3, factor_q=1).cuda()
x = torch.randn(512, 81, 17, 2).cuda()
ref_3D = torch.randn(512, 1, 17, 3).cuda()
ref_2D = torch.randn(1, 999, 17, 2).cuda()
seed = torch.randn(1024, 17, 3).cuda()
# t = TBlock(32*17, num_heads=8, mlp_ratio=2, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=norm_layer, down_factor=3).cuda()
p = PoseTransformerInverse(num_frame=81, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                                num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0, down_factor=3).cuda()
a = PoseTransformer(num_frame=81, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                                num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0).cuda()
g = PoseGroupTransformer(num_frame=81, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                                num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0, down_factor=3).cuda()
w = SwinTransformer(num_frame=81, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                                num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0, M=9, dilated_M=9).cuda()
s = ShiftTransformer(num_frame=81, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                                num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0, M=9, shift=3).cuda()
sg = ShiftGroupTransformer(num_frame=81, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                                num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0, M=9, shift=3).cuda()
m = MutualTransformer(num_frame=81, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                                num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0, M=9, shift=3, F=999).cuda()
e = ExtTransformer(num_frame=81, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None, M=9, shift=3, seed=seed).cuda()
noext = NoExt(num_frame=81, num_joints=17, in_chans=2,
                                          embed_dim_ratio=32, depth=4,
                                          num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                                          drop_path_rate=0.0, M=9, shift=3, seed=ref_3D, drop_rate=0.0).cuda()
nogroup = NoGroup(num_frame=81, num_joints=17, in_chans=2,
                                          embed_dim_ratio=32, depth=4,
                                          num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                                          drop_path_rate=0.0, M=9, shift=3, seed=ref_3D, drop_rate=0.0).cuda()
# interval = 0.0
# with torch.no_grad():
#     for i in range(50):
#         start = time.time()
#         y = t(x)
#         end = time.time()
#         interval = interval + end - start
#         print(end - start)
# interval = interval / 50
# print(interval)



with torch.no_grad():
    for i in range(50):
        start = time.time()
        # y = m(ref_3D, ref_2D, x)
        # y = a(x)
        y = nogroup(x)
        end = time.time()
        print(end-start)

model_params = 0
for parameter in nogroup.parameters():
    model_params += parameter.numel()
print('INFO: Trainable parameter count:', model_params)
