# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np

from common.arguments import parse_args
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno
import math

from einops import rearrange, repeat
from copy import deepcopy

from common.camera import *
import collections

from common.model_poseformer import *

from common.loss import *
from common.generators import ChunkedGenerator, UnchunkedGenerator, ChunkedDataset
from time import time
from common.utils import *
from tensorboardX import SummaryWriter
import shutil
from torch.utils.data import DataLoader
import random

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
# print(torch.cuda.device_count())

bone = {'S1': 0.25173345605532327, 'S5': 0.24862031986316044, 'S6': 0.2579145073890686, 'S7': 0.24729864050944647, 'S8': 0.24417704145113628, 'S9': 0.24901529302199682, 'S11': 0.24831129199471966}


###################
args = parse_args()
# print(args)

try:
    # Create checkpoint directory if it does not exist
    os.makedirs(args.checkpoint)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

log_path = f'./logs/{args.model_type}-{args.embed_ratio}'
if os.path.exists(log_path):
    shutil.rmtree(log_path)
os.mkdir(log_path)

writer = SummaryWriter(log_path)

print('Loading dataset...')
dataset_path = 'data/data_3d_' + args.dataset + '.npz'
if args.dataset == 'h36m':
    from common.h36m_dataset import Human36mDataset
    dataset = Human36mDataset(dataset_path)
elif args.dataset.startswith('humaneva'):
    from common.humaneva_dataset import HumanEvaDataset
    dataset = HumanEvaDataset(dataset_path)
elif args.dataset.startswith('custom'):
    from common.custom_dataset import CustomDataset
    dataset = CustomDataset('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz')
else:
    raise KeyError('Invalid dataset')

print('Preparing data...')
for subject in dataset.subjects():
    for action in dataset[subject].keys():
        anim = dataset[subject][action]

        if 'positions' in anim:
            positions_3d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                pos_3d[:, 1:] -= pos_3d[:, :1] # Remove global offset, but keep trajectory in first position

                positions_3d.append(pos_3d)
            anim['positions_3d'] = positions_3d

print('Loading 2D detections...')
keypoints = np.load('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz', allow_pickle=True)
keypoints_gt = np.load('data/data_2d_' + args.dataset + '_' + 'gt' + '.npz', allow_pickle=True)
keypoints_metadata = keypoints['metadata'].item()
keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
keypoints = keypoints['positions_2d'].item()
keypoints_gt = keypoints_gt['positions_2d'].item()

###################
for subject in dataset.subjects():
    assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
    for action in dataset[subject].keys():
        assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)
        if 'positions_3d' not in dataset[subject][action]:
            continue

        for cam_idx in range(len(keypoints[subject][action])):

            # We check for >= instead of == because some videos in H3.6M contain extra frames
            mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
            assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length
            assert keypoints_gt[subject][action][cam_idx].shape[0] >= mocap_length

            if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                # Shorten sequence
                keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]
            if keypoints_gt[subject][action][cam_idx].shape[0] > mocap_length:
                # Shorten sequence
                keypoints_gt[subject][action][cam_idx] = keypoints_gt[subject][action][cam_idx][:mocap_length]


        assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])
        assert len(keypoints_gt[subject][action]) == len(dataset[subject][action]['positions_3d'])

for subject in keypoints.keys():
    for action in keypoints[subject]:
        for cam_idx, kps in enumerate(keypoints[subject][action]):
            # Normalize camera frame
            cam = dataset.cameras()[subject][cam_idx]
            # print(cam['intrinsic'])
            # print(kps[0, :, :])
            kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])

            if args.k_norm:
                kps[..., :2] = normalize_K(kps[..., :2], K=cam['intrinsic'])

            keypoints[subject][action][cam_idx] = kps

for subject in keypoints_gt.keys():
    for action in keypoints_gt[subject]:
        for cam_idx, kps_gt in enumerate(keypoints_gt[subject][action]):
            # Normalize camera frame
            cam = dataset.cameras()[subject][cam_idx]
            # print(cam['intrinsic'])
            # print(kps[0, :, :])
            kps_gt[..., :2] = normalize_screen_coordinates(kps_gt[..., :2], w=cam['res_w'], h=cam['res_h'])

            if args.k_norm:
                kps_gt[..., :2] = normalize_K(kps_gt[..., :2], K=cam['intrinsic'])

            keypoints_gt[subject][action][cam_idx] = kps_gt

subjects_train = args.subjects_train.split(',')
subjects_semi = [] if not args.subjects_unlabeled else args.subjects_unlabeled.split(',')
if not args.render:
    subjects_test = args.subjects_test.split(',')
else:
    subjects_test = [args.viz_subject]


def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_poses_2d_gt = []
    out_camera_params = []
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]
            poses_2d_gt = keypoints_gt[subject][action]
            for i in range(len(poses_2d)): # Iterate across cameras
                out_poses_2d.append(poses_2d[i])
                out_poses_2d_gt.append(poses_2d_gt[i])

            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'])

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)): # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None

    stride = args.downsample
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i])//stride * subset)*stride)
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start+n_frames:stride]
            out_poses_2d_gt[i] = out_poses_2d_gt[i][start:start + n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start+n_frames:stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            out_poses_2d_gt[i] = out_poses_2d_gt[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]


    return out_camera_params, out_poses_3d, out_poses_2d, out_poses_2d_gt

action_filter = None if args.actions == '*' else args.actions.split(',')
if action_filter is not None:
    print('Selected actions:', action_filter)

cameras_valid, poses_valid, poses_valid_2d, poses_valid_2d_gt = fetch(subjects_test, action_filter, subset=0.1)


receptive_field = args.number_of_frames
print('INFO: Receptive field: {} frames'.format(receptive_field))
pad = (receptive_field -1) // 2 # Padding on each side
min_loss = 100000
width = cam['res_w']
height = cam['res_h']
num_joints = keypoints_metadata['num_joints']



#########################################PoseTransformer

if args.model_type == 'poseformer':
    model_pos_train = PoseTransformer(num_frame=receptive_field, num_joints=num_joints, in_chans=2,
                                embed_dim_ratio=args.embed_ratio, depth=4,
                                num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=args.droppath)
elif args.model_type == 'group':
    model_pos_train = PoseGroupTransformer(num_frame=receptive_field, num_joints=num_joints, in_chans=2, embed_dim_ratio=args.embed_ratio, depth=4,
            num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=args.droppath, down_factor=1, merge=args.merge, merge_type=args.merge_type, ST=args.ST, pos_type=args.pos_type, weighted=args.weighted)
elif args.model_type == 'share':
    model_pos_train = PoseGroupShareTransformer(num_frame=receptive_field, num_joints=num_joints, in_chans=2,
                                     embed_dim_ratio=args.embed_ratio, depth=4,
                                     num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0.1,
                                     down_factor=1, merge=args.merge)
elif args.model_type == 'refine':
    model_pos_train = PoseRefineTransformer(num_frame=receptive_field, num_joints=num_joints, in_chans=2,
                                           embed_dim_ratio=args.embed_ratio, depth=4,
                                           num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                                           drop_path_rate=args.droppath, down_factor=1, merge=args.merge,
                                           merge_type=args.merge_type, ST=args.ST, pos_type=args.pos_type,
                                           weighted=args.weighted)
elif args.model_type == 'swin':
    model_pos_train = SwinTransformer(num_frame=receptive_field, num_joints=num_joints, in_chans=2,
                                      embed_dim_ratio=args.embed_ratio, depth=4,
                                      num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                                      drop_path_rate=args.droppath, M=9, dilated_M=9)
elif args.model_type == 'shift':
    model_pos_train = ShiftTransformer(num_frame=receptive_field, num_joints=num_joints, in_chans=2,
                                      embed_dim_ratio=args.embed_ratio, depth=2,
                                      num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                                      drop_path_rate=args.droppath, M=9, shift=3)
elif args.model_type == 'shift-group':
    model_pos_train = ShiftGroupTransformer(num_frame=receptive_field, num_joints=num_joints, in_chans=2,
                                      embed_dim_ratio=args.embed_ratio, depth=4,
                                      num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=args.droppath,
                                      merge=args.merge, merge_type=args.merge_type, M=9, shift=3)
elif args.model_type == 'mutual':
    model_pos_train = MutualTransformer(num_frame=receptive_field, num_joints=num_joints, in_chans=2,
                                            embed_dim_ratio=args.embed_ratio, depth=2,
                                            num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                                            drop_path_rate=args.droppath,
                                            M=9, shift=3, F=999)
elif args.model_type == 'shift-group-dual':
    model_pos_train = ShiftGroupDualTransformer(num_frame=receptive_field, num_joints=num_joints, in_chans=2,
                                  embed_dim_ratio=args.embed_ratio, depth=4,
                                  num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                                  drop_path_rate=args.droppath,
                                  M=9, shift=3, F=999, merge=args.merge, merge_type=args.merge_type)
elif args.model_type == 'shift-group-dual-short':
    model_pos_train = ShiftGroupDualShortTransformer(num_frame=receptive_field, num_joints=num_joints, in_chans=2,
                                  embed_dim_ratio=args.embed_ratio, depth=4,
                                  num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                                  drop_path_rate=args.droppath,
                                  M=9, shift=3, F=999, merge=args.merge, merge_type=args.merge_type)
else:
    raise Exception('Wrong type of model!')

if args.model_type == 'poseformer':
    model_pos = PoseTransformer(num_frame=receptive_field, num_joints=num_joints, in_chans=2,
                                     embed_dim_ratio=args.embed_ratio, depth=4,
                                     num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0)
elif args.model_type == 'group':
    model_pos = PoseGroupTransformer(num_frame=receptive_field, num_joints=num_joints, in_chans=2,
                                embed_dim_ratio=args.embed_ratio, depth=4,
                                num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0, down_factor=1, merge=args.merge, merge_type=args.merge_type, ST=args.ST, pos_type=args.pos_type, weighted=args.weighted)
elif args.model_type == 'share':
    model_pos = PoseGroupShareTransformer(num_frame=receptive_field, num_joints=num_joints, in_chans=2,
                                     embed_dim_ratio=args.embed_ratio, depth=4,
                                     num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0,
                                     down_factor=1, merge=args.merge)
elif args.model_type == 'refine':
    model_pos = PoseRefineTransformer(num_frame=receptive_field, num_joints=num_joints, in_chans=2,
                                           embed_dim_ratio=args.embed_ratio, depth=4,
                                           num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                                           drop_path_rate=args.droppath, down_factor=1, merge=args.merge,
                                           merge_type=args.merge_type, ST=args.ST, pos_type=args.pos_type,
                                           weighted=args.weighted)
elif args.model_type == 'swin':
    model_pos = SwinTransformer(num_frame=receptive_field, num_joints=num_joints, in_chans=2,
                                embed_dim_ratio=args.embed_ratio, depth=4,
                                num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0, M=9, dilated_M=9)
elif args.model_type == 'shift':
    model_pos = ShiftTransformer(num_frame=receptive_field, num_joints=num_joints, in_chans=2,
                                embed_dim_ratio=args.embed_ratio, depth=2,
                                num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0, M=9, shift=3)
elif args.model_type == 'shift-group':
    model_pos = ShiftGroupTransformer(num_frame=receptive_field, num_joints=num_joints, in_chans=2,
                                embed_dim_ratio=args.embed_ratio, depth=4,
                                num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0, merge=args.merge, merge_type=args.merge_type, M=9, shift=3)
elif args.model_type == 'mutual':
    model_pos = MutualTransformer(num_frame=receptive_field, num_joints=num_joints, in_chans=2,
                                            embed_dim_ratio=args.embed_ratio, depth=2,
                                            num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                                            drop_path_rate=0,
                                            M=9, shift=3, F=999)
elif args.model_type == 'shift-group-dual':
    model_pos = ShiftGroupDualTransformer(num_frame=receptive_field, num_joints=num_joints, in_chans=2,
                                  embed_dim_ratio=args.embed_ratio, depth=4,
                                  num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                                  drop_path_rate=0,
                                  M=9, shift=3, F=999, merge=args.merge, merge_type=args.merge_type)
elif args.model_type == 'shift-group-dual-short':
    model_pos = ShiftGroupDualShortTransformer(num_frame=receptive_field, num_joints=num_joints, in_chans=2,
                                  embed_dim_ratio=args.embed_ratio, depth=4,
                                  num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                                  drop_path_rate=0,
                                  M=9, shift=3, F=999, merge=args.merge, merge_type=args.merge_type)
else:
    raise Exception('Wrong type of model!')


################ load weight ########################
# posetrans_checkpoint = torch.load('./checkpoint/pretrained_posetrans.bin', map_location=lambda storage, loc: storage)
# posetrans_checkpoint = posetrans_checkpoint["model_pos"]
# model_pos_train = load_pretrained_weights(model_pos_train, posetrans_checkpoint)

#################
causal_shift = 0
model_params = 0
for parameter in model_pos.parameters():
    model_params += parameter.numel()
print('INFO: Trainable parameter count:', model_params)

if torch.cuda.is_available():
    model_pos = nn.DataParallel(model_pos)
    model_pos = model_pos.cuda()
    model_pos_train = nn.DataParallel(model_pos_train)
    model_pos_train = model_pos_train.cuda()


if args.finetune:
    chk_filename = os.path.join(args.finetune)
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    checkpoint = checkpoint['model_pos']
    remove_keys = []
    pad_keys = []
    for k, v in checkpoint.items():
        if k.replace('module.', '').startswith('Temporal_pos_embed'):
            remove_keys.append(k)
        if k.replace('module.', '').startswith('weighted_mean'):
            pad_keys.append(k)
    for k in remove_keys:
        del checkpoint[k]
    for k in pad_keys:
        if len(checkpoint[k].shape) == 3:
            pad_w = (receptive_field - checkpoint[k].shape[1]) // 2
            pad_w = torch.zeros(1, pad_w, 1, dtype=torch.float32, device=checkpoint[k].device)
            checkpoint[k] = torch.cat([pad_w, checkpoint[k], pad_w], dim=1)

    model_pos_train.load_state_dict(checkpoint, strict=False)
    model_pos.load_state_dict(checkpoint, strict=False)


if args.resume or args.evaluate:
    chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    model_pos_train.load_state_dict(checkpoint['model_pos'], strict=False)
    model_pos.load_state_dict(checkpoint['model_pos'], strict=False)


test_generator = UnchunkedGenerator(cameras_valid, poses_valid, poses_valid_2d,
                                    pad=pad, causal_shift=causal_shift, augment=False,
                                    kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
print('INFO: Testing on {} frames'.format(test_generator.num_frames()))

def eval_data_prepare(receptive_field, inputs_2d, inputs_3d):
    inputs_2d_p = torch.squeeze(inputs_2d)
    inputs_3d_p = inputs_3d.permute(1,0,2,3)
    out_num = inputs_2d_p.shape[0] - receptive_field + 1
    eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
    for i in range(out_num):
        eval_input_2d[i,:,:,:] = inputs_2d_p[i:i+receptive_field, :, :]
    return eval_input_2d, inputs_3d_p


def attack(model, inputs_2d, inputs_3d, eps, attack_iters, alpha):
    model.eval()
    delta = torch.zeros_like(inputs_2d, dtype=inputs_2d.dtype, device=inputs_2d.device)
    delta.uniform_(-eps, eps)
    inputs_2d_adv = inputs_2d.detach() + delta

    for _ in range(attack_iters):
        inputs_2d_adv.requires_grad_()
        predicted_3d_pos = model_pos(inputs_2d_adv)
        error = mpjpe(predicted_3d_pos, inputs_3d)
        grad = torch.autograd.grad(error, [inputs_2d_adv])[0]
        grad = grad.detach()
        inputs_2d_adv = torch.max(torch.min(inputs_2d_adv.detach() + alpha * torch.sign(grad), inputs_2d + eps),
                                  inputs_2d - eps)
    return inputs_2d_adv



def prepare_actions_train():
    all_actions = {}
    all_actions_by_subject = {}
    for subject in subjects_train:
        if subject not in all_actions_by_subject:
            all_actions_by_subject[subject] = {}

        for action in dataset[subject].keys():
            action_name = action.split(' ')[0]
            if action_name not in all_actions:
                all_actions[action_name] = []
            if action_name not in all_actions_by_subject[subject]:
                all_actions_by_subject[subject][action_name] = []
            all_actions[action_name].append((subject, action))
            all_actions_by_subject[subject][action_name].append((subject, action))
    return all_actions, all_actions_by_subject

_, all_actions_by_subject_train = prepare_actions_train()


def evaluate_batch(test_generator, action=None, return_predictions=False, use_trajectory_model=False, out=False):
    epoch_loss_3d_pos = 0
    epoch_loss_3d_pos_procrustes = 0
    epoch_loss_3d_pos_scale = 0
    epoch_loss_3d_vel = 0
    with torch.no_grad():
        if not use_trajectory_model:
            model_pos.eval()
        # else:
        # model_traj.eval()
        N = 0
        for _, batch, batch_2d in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            inputs_3d = torch.from_numpy(batch.astype('float32'))



            ##### apply test-time-augmentation (following Videopose3d)
            inputs_2d_flip = inputs_2d.clone()
            inputs_2d_flip[:, :, :, 0] *= -1
            inputs_2d_flip[:, :, kps_left + kps_right, :] = inputs_2d_flip[:, :, kps_right + kps_left, :]

            ##### convert size
            inputs_2d, inputs_3d = eval_data_prepare(receptive_field, inputs_2d, inputs_3d)
            inputs_2d_flip, _ = eval_data_prepare(receptive_field, inputs_2d_flip, inputs_3d)

            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda(non_blocking=True)
                inputs_2d_flip = inputs_2d_flip.cuda(non_blocking=True)
                inputs_3d = inputs_3d.cuda(non_blocking=True)
            inputs_3d[:, :, 0] = 0

            flag = True
            while flag:
                SUBJECT = S_list[random.randint(0, len(S_list) - 1)]
                ACTION = Action_list[random.randint(0, len(Action_list) - 1)]
                poses_act, poses_2d_act = fetch_actions(all_actions_by_subject_train[SUBJECT][ACTION])
                if poses_act[0].shape[0] < 999:
                    continue
                else:
                    flag = False
                # ref_3D = torch.from_numpy(poses_act[0][:999]).float().unsqueeze(0).repeat(8, 1, 1, 1)
                # print(ref_3D.shape)
                # ref_3D[:, :, 0] = 0.0
                if args.model_type == 'shift-group-dual-short':
                    start = random.randint(0, poses_act[0].shape[0] - receptive_field)
                    ref_2D = torch.from_numpy(poses_2d_act[0][start:start + receptive_field]).float().unsqueeze(
                        0).repeat(inputs_2d.shape[0], 1, 1, 1)
                else:
                    ref_2D = torch.from_numpy(poses_2d_act[0][:999]).float().unsqueeze(0).repeat(8, 1, 1, 1)

                if torch.cuda.is_available():
                    # ref_3D = ref_3D.cuda(non_blocking=True)
                    ref_2D = ref_2D.cuda(non_blocking=True)

            if inputs_2d.shape[0] <= 2000:
                if args.model_type == 'mutual' or args.model_type == 'shift-group-dual':
                    predicted_3d_pos = model_pos(ref_2D, inputs_2d)
                    predicted_3d_pos_flip = model_pos(ref_2D, inputs_2d_flip)
                elif args.model_type == 'shift-group-dual-short':
                    predicted_3d_pos = model_pos(ref_2D, inputs_2d, eval=True)
                    predicted_3d_pos_flip = model_pos(ref_2D, inputs_2d_flip, eval=True)
                else:
                    predicted_3d_pos = model_pos(inputs_2d)
                    predicted_3d_pos_flip = model_pos(inputs_2d_flip)
            else:
                predicted_3d_pos = []
                predicted_3d_pos_flip = []
                if args.model_type == 'mutual' or args.model_type == 'shift-group-dual':
                    for kk in range(inputs_2d.shape[0] // 2000 + 1):
                        predicted_3d_pos.append(
                            model_pos(ref_2D, inputs_2d[kk * 2000: min(kk * 2000 + 2000, inputs_2d.shape[0]), ...]))
                        predicted_3d_pos_flip.append(
                            model_pos(ref_2D, inputs_2d_flip[kk * 2000: min(kk * 2000 + 2000, inputs_2d.shape[0]), ...]))
                elif args.model_type == 'shift-group-dual-short':
                    for kk in range(inputs_2d.shape[0] // 2000 + 1):
                        predicted_3d_pos.append(
                            model_pos(ref_2D[kk * 2000: min(kk * 2000 + 2000, inputs_2d.shape[0]), ...], inputs_2d[kk * 2000: min(kk * 2000 + 2000, inputs_2d.shape[0]), ...], eval=True))
                        predicted_3d_pos_flip.append(
                            model_pos(ref_2D[kk * 2000: min(kk * 2000 + 2000, inputs_2d.shape[0]), ...], inputs_2d_flip[kk * 2000: min(kk * 2000 + 2000, inputs_2d.shape[0]), ...], eval=True))
                else:
                    for kk in range(inputs_2d.shape[0] // 2000 + 1):
                        predicted_3d_pos.append(
                            model_pos(inputs_2d[kk * 2000: min(kk * 2000 + 2000, inputs_2d.shape[0]), ...]))
                        predicted_3d_pos_flip.append(
                            model_pos(inputs_2d_flip[kk * 2000: min(kk * 2000 + 2000, inputs_2d.shape[0]), ...]))
                predicted_3d_pos = torch.cat(predicted_3d_pos, dim=0)
                predicted_3d_pos_flip = torch.cat(predicted_3d_pos_flip, dim=0)

            predicted_3d_pos_flip[:, :, :, 0] *= -1
            predicted_3d_pos_flip[:, :, joints_left + joints_right] = predicted_3d_pos_flip[:, :,
                                                                      joints_right + joints_left]

            predicted_3d_pos = torch.mean(torch.cat((predicted_3d_pos, predicted_3d_pos_flip), dim=1), dim=1,
                                          keepdim=True)
            if args.bone_norm:
                bone_length = torch.sqrt(
                    torch.sum((inputs_3d[:, :, 12:13] - inputs_3d[:, :, 13:14]) ** 2, dim=-1, keepdim=True)).mean(1,
                                                                                                                  keepdim=True)
                predicted_3d_pos = predicted_3d_pos * bone_length

            # del inputs_2d, inputs_2d_flip
            # torch.cuda.empty_cache()

            if return_predictions:
                return predicted_3d_pos.squeeze(0).cpu().numpy()

            error = mpjpe(predicted_3d_pos, inputs_3d)
            epoch_loss_3d_pos_scale += inputs_3d.shape[0] * inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos,
                                                                                         inputs_3d).item()

            epoch_loss_3d_pos += inputs_3d.shape[0] * inputs_3d.shape[1] * error.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

            epoch_loss_3d_pos_procrustes += inputs_3d.shape[0] * inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)

            # Compute velocity error
            epoch_loss_3d_vel += inputs_3d.shape[0] * inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)

    e1 = (epoch_loss_3d_pos / N) * 1000
    e2 = (epoch_loss_3d_pos_procrustes / N) * 1000
    e3 = (epoch_loss_3d_pos_scale / N) * 1000
    ev = (epoch_loss_3d_vel / N) * 1000

    if out:
        if action is None:
            print('----------')
        else:
            print('----' + action + '----')
        print('Protocol #1 Error (MPJPE):', e1, 'mm')
        print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
        print('Protocol #3 Error (N-MPJPE):', e3, 'mm')
        print('Velocity Error (MPJVE):', ev, 'mm')
        print('----------')

    return e1, e2, e3, ev, N


# Evaluate
def evaluate2(test_generator, action=None, return_predictions=False, use_trajectory_model=False, out=False):
    epoch_loss_3d_pos = 0
    epoch_loss_3d_pos_procrustes = 0
    epoch_loss_3d_pos_scale = 0
    epoch_loss_3d_vel = 0
    with torch.no_grad():
        if not use_trajectory_model:
            model_pos.eval()
        # else:
            # model_traj.eval()
        N = 0
        for _, batch, batch_2d in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            inputs_3d = torch.from_numpy(batch.astype('float32'))


            ##### apply test-time-augmentation (following Videopose3d)
            inputs_2d_flip = inputs_2d.clone()
            inputs_2d_flip [:, :, :, 0] *= -1
            inputs_2d_flip[:, :, kps_left + kps_right,:] = inputs_2d_flip[:, :, kps_right + kps_left,:]

            ##### convert size
            inputs_2d, inputs_3d = eval_data_prepare(receptive_field, inputs_2d, inputs_3d)
            inputs_2d_flip, _ = eval_data_prepare(receptive_field, inputs_2d_flip, inputs_3d)

            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda(non_blocking=True)
                inputs_2d_flip = inputs_2d_flip.cuda(non_blocking=True)
                inputs_3d = inputs_3d.cuda(non_blocking=True)
            inputs_3d[:, :, 0] = 0

            if args.model_type == 'mutual' or args.model_type == 'shift-group-dual':
                predicted_3d_pos = model_pos(ref_2D, inputs_2d)
                predicted_3d_pos_flip = model_pos(ref_2D, inputs_2d_flip)
            else:
                predicted_3d_pos = model_pos(inputs_2d)
                predicted_3d_pos_flip = model_pos(inputs_2d_flip)

            predicted_3d_pos_flip[:, :, :, 0] *= -1
            predicted_3d_pos_flip[:, :, joints_left + joints_right] = predicted_3d_pos_flip[:, :,
                                                                      joints_right + joints_left]

            predicted_3d_pos = torch.mean(torch.cat((predicted_3d_pos, predicted_3d_pos_flip), dim=1), dim=1,
                                          keepdim=True)
            if args.bone_norm:
                bone_length = torch.sqrt(
                    torch.sum((inputs_3d[:, :, 12:13] - inputs_3d[:, :, 13:14]) ** 2, dim=-1, keepdim=True)).mean(1, keepdim=True)
                predicted_3d_pos = predicted_3d_pos * bone_length

            # del inputs_2d, inputs_2d_flip
            # torch.cuda.empty_cache()

            if return_predictions:
                return predicted_3d_pos.squeeze(0).cpu().numpy()


            error = mpjpe(predicted_3d_pos, inputs_3d)
            epoch_loss_3d_pos_scale += inputs_3d.shape[0]*inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos, inputs_3d).item()

            epoch_loss_3d_pos += inputs_3d.shape[0]*inputs_3d.shape[1] * error.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

            epoch_loss_3d_pos_procrustes += inputs_3d.shape[0]*inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)

            # Compute velocity error
            epoch_loss_3d_vel += inputs_3d.shape[0]*inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)

    e1 = (epoch_loss_3d_pos / N)*1000
    e2 = (epoch_loss_3d_pos_procrustes / N)*1000
    e3 = (epoch_loss_3d_pos_scale / N)*1000
    ev = (epoch_loss_3d_vel / N)*1000
    if out:
        if action is None:
            print('----------')
        else:
            print('----'+action+'----')
        print('Protocol #1 Error (MPJPE):', e1, 'mm')
        print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
        print('Protocol #3 Error (N-MPJPE):', e3, 'mm')
        print('Velocity Error (MPJVE):', ev, 'mm')
        print('----------')

    return e1, e2, e3, ev, N


def attack_eval(test_generator, action=None, return_predictions=False, use_trajectory_model=False, eps=0.01, attack_iters=10, alpha=0.0025):
    epoch_loss_3d_pos = 0
    epoch_loss_3d_pos_procrustes = 0
    epoch_loss_3d_pos_scale = 0
    epoch_loss_3d_vel = 0
    model_pos.eval()
    N = 0

    for _, batch, batch_2d in test_generator.next_epoch():
        inputs_2d_total = torch.from_numpy(batch_2d.astype('float32'))
        inputs_3d_total = torch.from_numpy(batch.astype('float32'))
        if torch.cuda.is_available():
            inputs_2d_total = inputs_2d_total.cuda(non_blocking=True)
            inputs_3d_total = inputs_3d_total.cuda(non_blocking=True)

        ##### convert size

        inputs_2d_total, inputs_3d_total = eval_data_prepare(receptive_field, inputs_2d_total, inputs_3d_total)

        # inputs_2d_flip, _ = eval_data_prepare(receptive_field, inputs_2d_flip, inputs_3d)

        inputs_3d_total[:, :, 0] = 0


        for iter in range(11):

            if iter < 10:
                inputs_2d = inputs_2d_total[iter*(inputs_2d_total.shape[0] // 10):(iter+1)*(inputs_2d_total.shape[0] // 10)]
                inputs_3d = inputs_3d_total[
                            iter * (inputs_3d_total.shape[0] // 10):(iter + 1) * (inputs_3d_total.shape[0] // 10)]
            else:
                inputs_2d = inputs_2d_total[
                            10 * (inputs_2d_total.shape[0] // 10):]
                inputs_3d = inputs_3d_total[
                            10 * (inputs_3d_total.shape[0] // 10):]



            delta = torch.zeros_like(inputs_2d, dtype=inputs_2d.dtype, device=inputs_2d.device)
            delta.uniform_(-eps, eps)
            inputs_2d_adv = inputs_2d.detach() + delta


            for _ in range(attack_iters):
                inputs_2d_adv.requires_grad_()
                inputs_2d_flip_adv = inputs_2d_adv.clone()
                inputs_2d_flip_adv[:, :, :, 0] *= -1
                inputs_2d_flip_adv[:, :, kps_left + kps_right, :] = inputs_2d_flip_adv[:, :, kps_right + kps_left, :]

                predicted_3d_pos = model_pos(inputs_2d_adv)
                predicted_3d_pos_flip = model_pos(inputs_2d_flip_adv)
                predicted_3d_pos_flip[:, :, :, 0] *= -1
                predicted_3d_pos_flip[:, :, joints_left + joints_right] = predicted_3d_pos_flip[:, :, joints_right + joints_left]

                predicted_3d_pos = torch.mean(torch.cat((predicted_3d_pos, predicted_3d_pos_flip), dim=1), dim=1,
                                                    keepdim=True)


                error = mpjpe(predicted_3d_pos, inputs_3d)
                grad = torch.autograd.grad(error, [inputs_2d_adv])[0]
                grad = grad.detach()
                inputs_2d_adv = torch.max(torch.min(inputs_2d_adv.detach() + alpha * torch.sign(grad), inputs_2d + eps), inputs_2d -eps)

            inputs_2d_adv = inputs_2d_adv.detach()
            inputs_2d_flip_adv = inputs_2d_adv.clone()
            inputs_2d_flip_adv[:, :, :, 0] *= -1
            inputs_2d_flip_adv[:, :, kps_left + kps_right, :] = inputs_2d_flip_adv[:, :, kps_right + kps_left, :]

            with torch.no_grad():
                predicted_3d_pos = model_pos(inputs_2d_adv)
                predicted_3d_pos_flip = model_pos(inputs_2d_flip_adv)
                predicted_3d_pos_flip[:, :, :, 0] *= -1
                predicted_3d_pos_flip[:, :, joints_left + joints_right] = predicted_3d_pos_flip[:, :,
                                                                          joints_right + joints_left]

            predicted_3d_pos = torch.mean(torch.cat((predicted_3d_pos, predicted_3d_pos_flip), dim=1), dim=1,
                                          keepdim=True)

            error = mpjpe(predicted_3d_pos, inputs_3d)

            epoch_loss_3d_pos_scale += inputs_3d.shape[0] * inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos, inputs_3d).item()

            epoch_loss_3d_pos += inputs_3d.shape[0] * inputs_3d.shape[1] * error.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

            epoch_loss_3d_pos_procrustes += inputs_3d.shape[0] * inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)

            # Compute velocity error
            epoch_loss_3d_vel += inputs_3d.shape[0] * inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)

    if action is None:
        print('----------')
    else:
        print('----' + action + '----')
    e1 = (epoch_loss_3d_pos / N) * 1000
    e2 = (epoch_loss_3d_pos_procrustes / N) * 1000
    e3 = (epoch_loss_3d_pos_scale / N) * 1000
    ev = (epoch_loss_3d_vel / N) * 1000
    print('Protocol #1 Adv Error (MPJPE):', e1, 'mm')
    print('Protocol #2 Adv Error (P-MPJPE):', e2, 'mm')
    print('Protocol #3 Adv Error (N-MPJPE):', e3, 'mm')
    print('Adv Velocity Error (MPJVE):', ev, 'mm')
    print('----------')

    return e1, e2, e3, ev, N

def fetch_actions(actions):
    out_poses_3d = []
    out_poses_2d = []

    for subject, action in actions:
        poses_2d = keypoints[subject][action]
        for i in range(len(poses_2d)):  # Iterate across cameras
            out_poses_2d.append(poses_2d[i])

        poses_3d = dataset[subject][action]['positions_3d']
        assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
        for i in range(len(poses_3d)):  # Iterate across cameras
            out_poses_3d.append(poses_3d[i])

    stride = args.downsample
    if stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]

    return out_poses_3d, out_poses_2d


def prepare_actions():
    all_actions = {}
    all_actions_by_subject = {}
    for subject in subjects_test:
        if subject not in all_actions_by_subject:
            all_actions_by_subject[subject] = {}

        for action in dataset[subject].keys():
            action_name = action.split(' ')[0]
            if action_name not in all_actions:
                all_actions[action_name] = []
            if action_name not in all_actions_by_subject[subject]:
                all_actions_by_subject[subject][action_name] = []
            all_actions[action_name].append((subject, action))
            all_actions_by_subject[subject][action_name].append((subject, action))
    return all_actions, all_actions_by_subject






def evaluate(test_generator, action=None, return_predictions=False, out=False):
    epoch_loss_3d_pos = 0
    epoch_loss_3d_pos_procrustes = 0
    epoch_loss_3d_pos_scale = 0
    epoch_loss_3d_vel = 0
    with torch.no_grad():
        model_pos.eval()
        N = 0
        for _, batch, batch_2d in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()

            # Positional model
            if batch is not None:
              inputs_3d = torch.from_numpy(batch.astype('float32'))
              if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
              inputs_3d[:, :, 0] = 0

            if test_generator.augment_enabled():
                inputs_2d_0, inputs_3d = eval_data_prepare(receptive_field, inputs_2d[0, ...], inputs_3d)
                inputs_2d_1, _ = eval_data_prepare(receptive_field, inputs_2d[1, ...], inputs_3d)
                inputs_2d = torch.cat([inputs_2d_0, inputs_2d_1], 0)
            else:
                inputs_2d, inputs_3d = eval_data_prepare(receptive_field, inputs_2d, inputs_3d)

            if args.model_type == 'mutual' or args.model_type == 'shift-group-dual':
                predicted_3d_pos = model_pos(ref_2D, inputs_2d)
            else:
                predicted_3d_pos = model_pos(inputs_2d)

            if test_generator.augment_enabled():
              if batch is not None:
                inputs_3d = inputs_3d[:, :1]
            if test_generator.augment_enabled():
                # Undo flipping and take average with non-flipped version
                predicted_3d_pos = predicted_3d_pos.view(2, -1, *predicted_3d_pos.shape[2:])
                predicted_3d_pos[1, :, :, 0] *= -1
                predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
                predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)
                predicted_3d_pos = predicted_3d_pos.transpose(0, 1)

            if return_predictions:
                return predicted_3d_pos.squeeze(0).cpu().numpy()
            pad = int((inputs_3d.size(0) - predicted_3d_pos.size(0))/2)
            inputs_3d = inputs_3d[pad:inputs_3d.size(0)-pad]
            error = mpjpe(predicted_3d_pos, inputs_3d)
            epoch_loss_3d_pos_scale += inputs_3d.shape[0]*inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos, inputs_3d).item()

            epoch_loss_3d_pos += inputs_3d.shape[0]*inputs_3d.shape[1] * error.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

            epoch_loss_3d_pos_procrustes += inputs_3d.shape[0]*inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)
            # Compute velocity error
            epoch_loss_3d_vel += inputs_3d.shape[0] * inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)

        e1 = (epoch_loss_3d_pos / N) * 1000
        e2 = (epoch_loss_3d_pos_procrustes / N) * 1000
        e3 = (epoch_loss_3d_pos_scale / N) * 1000
        ev = (epoch_loss_3d_vel / N) * 1000
        if out:
            if action is None:
                print('----------')
            else:
                print('----' + action + '----')
            print('Test time augmentation:', test_generator.augment_enabled())
            print('Protocol #1 Error (MPJPE):', e1, 'mm')
            print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
            print('Protocol #3 Error (N-MPJPE):', e3, 'mm')
            print('Velocity Error (MPJVE):', ev, 'mm')
            print('----------')

        return e1, e2, e3, ev, N


def run_evaluation(actions, action_filter=None, out=False):
    errors_p1 = []
    errors_p2 = []
    errors_p3 = []
    errors_vel = []
    Ns = []

    for action_key in actions.keys():
        if action_filter is not None:
            found = False
            for a in action_filter:
                if action_key.startswith(a):
                    found = True
                    break
            if not found:
                continue

        poses_act, poses_2d_act = fetch_actions(actions[action_key])
        gen = UnchunkedGenerator(None, poses_act, poses_2d_act,
                                 pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                                 kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                 joints_right=joints_right)
        e1, e2, e3, ev, N = evaluate_batch(gen, action_key, out=out)
        # e1_adv, e2_adv, e3_adv, ev_adv, N = attack_eval(gen, action_key)
        errors_p1.append(e1)
        errors_p2.append(e2)
        errors_p3.append(e3)
        errors_vel.append(ev)
        Ns.append(N)

    print('Protocol #1   (MPJPE) action-wise average:', round(np.mean(errors_p1), 1), 'mm')
    print('Protocol #2 (P-MPJPE) action-wise average:', round(np.mean(errors_p2), 1), 'mm')
    print('Protocol #3 (N-MPJPE) action-wise average:', round(np.mean(errors_p3), 1), 'mm')
    print('Velocity      (MPJVE) action-wise average:', round(np.mean(errors_vel), 2), 'mm')

    print('Protocol #1   (MPJPE) frame-wise average:', round(np.sum(np.array(errors_p1) * np.array(Ns)) / np.sum(Ns), 1), 'mm')
    print('Protocol #2 (P-MPJPE) frame-wise average:', round(np.sum(np.array(errors_p2) * np.array(Ns)) / np.sum(Ns), 1), 'mm')
    print('Protocol #3 (N-MPJPE) frame-wise average:', round(np.sum(np.array(errors_p3) * np.array(Ns)) / np.sum(Ns), 1), 'mm')
    print('Velocity      (MPJVE) frame-wise average:', round(np.sum(np.array(errors_vel) * np.array(Ns)) / np.sum(Ns), 1), 'mm')

###################

S_list = ['S1','S5','S6', 'S7','S8']
Action_list = ['Directions', 'Photo', 'Discussion', 'Walking',
               'Purchases', 'Phoning', 'Eating', 'Sitting', 'Walking', 'WalkDog', 'Waiting',
               'Posing', 'Greeting', 'Smoking', 'SittingDown']

if not args.evaluate:
    cameras_train, poses_train, poses_train_2d, poses_train_2d_gt = fetch(subjects_train, action_filter, subset=args.subset)

    lr = args.learning_rate
    optimizer = optim.AdamW(model_pos_train.parameters(), lr=lr, weight_decay=0.1)

    lr_decay = args.lr_decay
    losses_3d_train = []
    losses_3d_train_eval = []
    losses_3d_valid = []
    losses_3d_train.append(0.0)

    epoch = 0
    initial_momentum = 0.1
    final_momentum = 0.001
    if args.torch_loader:
        train_dataset = ChunkedDataset(cameras_train, poses_train, poses_train_2d,
                                           args.stride,
                                           pad=pad, causal_shift=causal_shift,
                                           augment=args.data_augmentation,
                                           kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                           joints_right=joints_right)
        train_generator = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.num_workers, pin_memory=True)
    else:
        if True:
            train_generator = ChunkedGenerator(args.batch_size//args.stride, cameras_train, poses_train, poses_train_2d, poses_train_2d_gt, args.stride,
                                                pad=pad, causal_shift=causal_shift, shuffle=True, augment=args.data_augmentation,
                                                kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
        else:
            train_generator = ChunkedGenerator(args.batch_size // args.stride, cameras_train, poses_train,
                                               poses_train_2d, None, args.stride,
                                               pad=pad, causal_shift=causal_shift, shuffle=True,
                                               augment=args.data_augmentation,
                                               kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                               joints_right=joints_right)

    train_generator_eval = UnchunkedGenerator(cameras_train, poses_train, poses_train_2d,
                                              pad=pad, causal_shift=causal_shift, augment=False)
    print('INFO: Training on {} frames'.format(train_generator_eval.num_frames()))


    if args.resume:
        epoch = checkpoint['epoch']
        if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            if not args.torch_loader:
                train_generator.set_random_state(checkpoint['random_state'])
        else:
            print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')

        # lr = checkpoint['lr']


    print('** Note: reported losses are averaged over all frames.')
    print('** The final evaluation will be carried out after the last training epoch.')

    # Pos model only
    loss_print = AverageMeter()
    if args.inter:
        loss_print_inter = AverageMeter()
    total_count = 0
    while epoch < args.epochs:
        start_time = time()
        epoch_loss_3d_train = 0
        epoch_loss_traj_train = 0
        epoch_loss_2d_train_unlabeled = 0
        N = 0
        N_semi = 0
        model_pos_train.train()
        loss_print.reset()
        if args.inter:
            loss_print_inter.reset()

        batch_count = 0
        if True:
            if args.torch_loader:
                for _, (cameras_train, batch_3d, batch_2d) in enumerate(train_generator):

                    cameras_train = cameras_train.float()
                    inputs_3d = batch_3d.float()
                    inputs_2d = batch_2d.float()

                    if torch.cuda.is_available():
                        inputs_3d = inputs_3d.cuda(non_blocking=True)
                        inputs_2d = inputs_2d.cuda(non_blocking=True)
                        cameras_train = cameras_train.cuda(non_blocking=True)
                    inputs_traj = inputs_3d[:, :, :1].clone()
                    inputs_3d[:, :, 0] = 0

                    optimizer.zero_grad()

                    # Predict 3D poses
                    if args.inter:
                        predicted_3d_pos, predicted_3d_pos_int = model_pos_train(inputs_2d, int=True)
                    else:
                        if args.model_type == 'refine':
                            predicted_3d_pos, predicted_2d_pos = model_pos_train(inputs_2d)
                        else:
                            predicted_3d_pos = model_pos_train(inputs_2d)

                    if args.bone_norm:
                        bone_length = torch.sqrt(
                            torch.sum((inputs_3d[:, :, 12:13] - inputs_3d[:, :, 13:14]) ** 2, dim=-1,
                                      keepdim=True)).mean(1, keepdim=True)
                        predicted_3d_pos = predicted_3d_pos * bone_length

                    # del inputs_2d
                    # torch.cuda.empty_cache()

                    loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                    if args.inter:
                        inputs_3d_inter = inputs_3d.clone()
                        inputs_3d_inter[:, :, (11, 12, 13, 14, 15, 16), :] = inputs_3d_inter[:, :,
                                                                             (11, 12, 13, 14, 15, 16),
                                                                             :] - inputs_3d_inter[:, :, 8:9, :]
                        loss_3d_pos_inter = mpjpe(predicted_3d_pos_int, inputs_3d_inter)
                        loss_print_inter.update(loss_3d_pos_inter.item())
                    loss_print.update(loss_3d_pos.item())
                    epoch_loss_3d_train += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
                    N += inputs_3d.shape[0] * inputs_3d.shape[1]

                    if args.inter:
                        loss_total = loss_3d_pos + loss_3d_pos_inter
                    else:
                        loss_total = loss_3d_pos

                    loss_total.backward()

                    optimizer.step()
                    # del inputs_3d, loss_3d_pos, predicted_3d_pos
                    # torch.cuda.empty_cache()
                    # if batch_count % 100 == 0:
                    #     if args.inter:
                    #         print(
                    #             f'Train epoch {epoch} / batch: {batch_count}: {loss_print.avg}, {loss_print_inter.avg}')
                    #     else:
                    #         print(f'Train epoch {epoch} / batch: {batch_count}: {loss_print.avg}')
                    if total_count % 100 == 0:
                        writer.add_scalar('train_loss', loss_print.val, global_step=total_count)
                    batch_count += 1
                    total_count += 1
            else:
                for cameras_train, batch_3d, batch_2d, batch_2d_gt in train_generator.next_epoch():
                    cameras_train = torch.from_numpy(cameras_train.astype('float32'))
                    inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
                    inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                    inputs_2d_gt = torch.from_numpy(batch_2d_gt.astype('float32'))
                    flag = True
                    while flag:
                        SUBJECT = S_list[random.randint(0, len(S_list)-1)]
                        ACTION = Action_list[random.randint(0, len(Action_list)-1)]
                        poses_act, poses_2d_act = fetch_actions(all_actions_by_subject_train[SUBJECT][ACTION])
                        if poses_act[0].shape[0] < 999:
                            continue
                        else:
                            flag = False
                        # ref_3D = torch.from_numpy(poses_act[0][:999]).float().unsqueeze(0).repeat(8, 1, 1, 1)
                        # print(ref_3D.shape)
                        # ref_3D[:, :, 0] = 0.0
                        ref_2D = torch.from_numpy(poses_2d_act[0][:999]).float().unsqueeze(0).repeat(8, 1, 1, 1)

                        if torch.cuda.is_available():
                            # ref_3D = ref_3D.cuda(non_blocking=True)
                            ref_2D = ref_2D.cuda(non_blocking=True)

                    if torch.cuda.is_available():
                        inputs_3d = inputs_3d.cuda(non_blocking=True)
                        inputs_2d = inputs_2d.cuda(non_blocking=True)
                        inputs_2d_gt = inputs_2d_gt.cuda(non_blocking=True)
                        cameras_train = cameras_train.cuda(non_blocking=True)
                    inputs_traj = inputs_3d[:, :, :1].clone()
                    inputs_3d[:, :, 0] = 0


                    if args.attack:
                        inputs_2d_adv = attack(model_pos_train, inputs_2d, inputs_3d, eps=0.01, attack_iters=1, alpha=0.0125)
                        model_pos_train.train()


                    optimizer.zero_grad()

                    # Predict 3D poses
                    if args.inter:
                        predicted_3d_pos, predicted_3d_pos_int = model_pos_train(inputs_2d, int=True)
                    else:
                        if args.model_type == 'refine':
                            predicted_3d_pos, predicted_2d_pos = model_pos_train(inputs_2d, refine=True)
                        elif args.model_type == 'mutual' or args.model_type == 'shift-group-dual':
                            predicted_3d_pos = model_pos_train(ref_2D, inputs_2d)
                        elif args.model_type == 'shift-group-dual-short':
                            predicted_3d_pos = model_pos_train(inputs_2d[:inputs_2d.shape[0]//2, ...], inputs_2d[inputs_2d.shape[0]//2:, ...])
                        else:
                            predicted_3d_pos = model_pos_train(inputs_2d)

                    if args.attack:
                        predicted_3d_pos_adv = model_pos_train(inputs_2d_adv)

                    if args.bone_norm:
                        bone_length = torch.sqrt(torch.sum((inputs_3d[:, :, 12:13] - inputs_3d[:, :, 13:14])**2, dim=-1, keepdim=True)).mean(1, keepdim=True)
                        predicted_3d_pos = predicted_3d_pos * bone_length

                    # del inputs_2d
                    # torch.cuda.empty_cache()
                    loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                    if args.attack:
                        loss_3d_pos_adv = mpjpe(predicted_3d_pos_adv, inputs_3d)

                    if args.inter:
                        inputs_3d_inter = inputs_3d.clone()
                        inputs_3d_inter[:, :, (11,12,13,14,15,16), :] = inputs_3d_inter[:, :, (11,12,13,14,15,16), :] - inputs_3d_inter[:, :, 8:9, :]
                        loss_3d_pos_inter = mpjpe(predicted_3d_pos_int, inputs_3d_inter)
                        loss_print_inter.update(loss_3d_pos_inter.item())
                    loss_print.update(loss_3d_pos.item())
                    epoch_loss_3d_train += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
                    N += inputs_3d.shape[0] * inputs_3d.shape[1]

                    if args.inter:
                        loss_total = loss_3d_pos + loss_3d_pos_inter
                    else:
                        if args.model_type == 'refine':
                            loss_2d = torch.sqrt(((predicted_2d_pos - inputs_2d) ** 2).sum(-1)).mean()
                            loss_total = loss_3d_pos + loss_2d
                        else:
                            loss_total = loss_3d_pos
                    if args.attack:
                        loss_total = loss_total + loss_3d_pos_adv

                    loss_total.backward()

                    optimizer.step()
                    # del inputs_3d, loss_3d_pos, predicted_3d_pos
                    # torch.cuda.empty_cache()
                    # if batch_count % 100 == 0:
                    #     if args.inter:
                    #         print(f'Train epoch {epoch} / batch: {batch_count}: {loss_print.avg}, {loss_print_inter.avg}')
                    #     else:
                    #         print(f'Train epoch {epoch} / batch: {batch_count}: {loss_print.avg}')
                    if total_count % 100 == 0:
                        writer.add_scalar('train_loss', loss_print.val, global_step=total_count)
                    batch_count += 1
                    total_count += 1

            losses_3d_train.append(epoch_loss_3d_train / N)
            # torch.cuda.empty_cache()

        # End-of-epoch evaluation
        with torch.no_grad():
            model_pos.load_state_dict(model_pos_train.state_dict(), strict=False)
            model_pos.eval()

            epoch_loss_3d_valid = 0
            epoch_loss_traj_valid = 0
            epoch_loss_2d_valid = 0
            N = 0
            loss_print.reset()
            if not args.no_eval:
                # Evaluate on test set
                batch_count = 0
                for cam, batch, batch_2d in test_generator.next_epoch():
                    inputs_3d = torch.from_numpy(batch.astype('float32'))
                    inputs_2d = torch.from_numpy(batch_2d.astype('float32'))

                    ##### apply test-time-augmentation (following Videopose3d)
                    inputs_2d_flip = inputs_2d.clone()
                    inputs_2d_flip[:, :, :, 0] *= -1
                    inputs_2d_flip[:, :, kps_left + kps_right, :] = inputs_2d_flip[:, :, kps_right + kps_left, :]

                    ##### convert size
                    inputs_2d, inputs_3d = eval_data_prepare(receptive_field, inputs_2d, inputs_3d)
                    inputs_2d_flip, _ = eval_data_prepare(receptive_field, inputs_2d_flip, inputs_3d)

                    if torch.cuda.is_available():
                        inputs_2d = inputs_2d.cuda(non_blocking=True)
                        inputs_2d_flip = inputs_2d_flip.cuda(non_blocking=True)
                        inputs_3d = inputs_3d.cuda(non_blocking=True)
                    inputs_3d[:, :, 0] = 0
                    # print(inputs_3d.shape[0])
                    flag = True
                    while flag:
                        SUBJECT = S_list[random.randint(0, len(S_list) - 1)]
                        ACTION = Action_list[random.randint(0, len(Action_list) - 1)]
                        poses_act, poses_2d_act = fetch_actions(all_actions_by_subject_train[SUBJECT][ACTION])
                        if poses_act[0].shape[0] < 999:
                            continue
                        else:
                            flag = False
                        # ref_3D = torch.from_numpy(poses_act[0][:999]).float().unsqueeze(0).repeat(8, 1, 1, 1)
                        # print(ref_3D.shape)
                        # ref_3D[:, :, 0] = 0.0

                        if args.model_type == 'shift-group-dual-short':
                            start = random.randint(0, poses_act[0].shape[0] - receptive_field)
                            ref_2D = torch.from_numpy(poses_2d_act[0][start:start + receptive_field]).float().unsqueeze(
                                0).repeat(inputs_2d.shape[0], 1, 1, 1)
                        else:
                            ref_2D = torch.from_numpy(poses_2d_act[0][:999]).float().unsqueeze(0).repeat(8, 1, 1, 1)

                        if torch.cuda.is_available():
                            # ref_3D = ref_3D.cuda(non_blocking=True)
                            ref_2D = ref_2D.cuda(non_blocking=True)

                    if inputs_2d.shape[0] <= 2000:
                        if args.model_type == 'mutual' or args.model_type == 'shift-group-dual':
                            predicted_3d_pos = model_pos(ref_2D, inputs_2d)
                        elif args.model_type == 'shift-group-dual-short':
                            predicted_3d_pos = model_pos(ref_2D, inputs_2d, eval=True)
                        else:
                            predicted_3d_pos = model_pos(inputs_2d)
                        if args.bone_norm:
                            bone_length = torch.sqrt(
                                torch.sum((inputs_3d[:, :, 12:13] - inputs_3d[:, :, 13:14]) ** 2, dim=-1, keepdim=True)).mean(1, keepdim=True)
                            predicted_3d_pos = predicted_3d_pos * bone_length
                        if args.model_type == 'mutual' or args.model_type == 'shift-group-dual':
                            predicted_3d_pos_flip = model_pos(ref_2D, inputs_2d_flip)
                        elif args.model_type == 'shift-group-dual-short':
                            predicted_3d_pos_flip = model_pos(ref_2D, inputs_2d_flip, eval=True)
                        else:
                            predicted_3d_pos_flip = model_pos(inputs_2d_flip)
                        if args.bone_norm:
                            bone_length = torch.sqrt(
                                torch.sum((inputs_3d[:, :, 12:13] - inputs_3d[:, :, 13:14]) ** 2, dim=-1, keepdim=True)).mean(1, keepdim=True)
                            predicted_3d_pos_flip = predicted_3d_pos_flip * bone_length
                    else:
                        predicted_3d_pos = []
                        predicted_3d_pos_flip = []
                        if args.model_type == 'mutual'  or args.model_type == 'shift-group-dual':
                            for kk in range(inputs_2d.shape[0] // 2000 + 1):
                                predicted_3d_pos.append(
                                    model_pos(ref_2D, inputs_2d[kk * 2000: min(kk * 2000 + 2000, inputs_2d.shape[0]), ...]))
                                predicted_3d_pos_flip.append(model_pos(ref_2D,
                                    inputs_2d_flip[kk * 2000: min(kk * 2000 + 2000, inputs_2d.shape[0]), ...]))
                        elif args.model_type == 'shift-group-dual-short':
                            for kk in range(inputs_2d.shape[0] // 2000 + 1):
                                predicted_3d_pos.append(
                                    model_pos(ref_2D[kk * 2000: min(kk * 2000 + 2000, inputs_2d.shape[0]), ...], inputs_2d[kk * 2000: min(kk * 2000 + 2000, inputs_2d.shape[0]), ...],eval=True))
                                predicted_3d_pos_flip.append(model_pos(ref_2D[kk * 2000: min(kk * 2000 + 2000, inputs_2d.shape[0]), ...],
                                    inputs_2d_flip[kk * 2000: min(kk * 2000 + 2000, inputs_2d.shape[0]), ...], eval=True))
                        else:
                            for kk in range(inputs_2d.shape[0] // 2000 + 1):
                                predicted_3d_pos.append(model_pos(inputs_2d[kk * 2000: min(kk * 2000 + 2000, inputs_2d.shape[0]), ...]))
                                predicted_3d_pos_flip.append(model_pos(inputs_2d_flip[kk * 2000: min(kk * 2000 + 2000, inputs_2d.shape[0]), ...]))
                        predicted_3d_pos = torch.cat(predicted_3d_pos, dim=0)
                        predicted_3d_pos_flip = torch.cat(predicted_3d_pos_flip, dim=0)
                        if args.bone_norm:
                            bone_length = torch.sqrt(
                                torch.sum((inputs_3d[:, :, 12:13] - inputs_3d[:, :, 13:14]) ** 2, dim=-1, keepdim=True)).mean(1, keepdim=True)
                            predicted_3d_pos = predicted_3d_pos * bone_length
                        if args.bone_norm:
                            bone_length = torch.sqrt(
                                torch.sum((inputs_3d[:, :, 12:13] - inputs_3d[:, :, 13:14]) ** 2, dim=-1, keepdim=True)).mean(1, keepdim=True)
                            predicted_3d_pos_flip = predicted_3d_pos_flip * bone_length

                    predicted_3d_pos_flip[:, :, :, 0] *= -1
                    predicted_3d_pos_flip[:, :, joints_left + joints_right] = predicted_3d_pos_flip[:, :,
                                                                              joints_right + joints_left]

                    predicted_3d_pos = torch.mean(torch.cat((predicted_3d_pos, predicted_3d_pos_flip), dim=1), dim=1,
                                                  keepdim=True)

                    # del inputs_2d, inputs_2d_flip
                    # torch.cuda.empty_cache()

                    loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                    loss_print.update(loss_3d_pos.item())
                    epoch_loss_3d_valid += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
                    N += inputs_3d.shape[0] * inputs_3d.shape[1]

                    # del inputs_3d, loss_3d_pos, predicted_3d_pos
                    # torch.cuda.empty_cache()
                    # if batch_count % 100 == 0:
                    #     print(f'Test epoch {epoch} / batch: {batch_count}: {loss_print.avg}')
                    batch_count += 1


                losses_3d_valid.append(epoch_loss_3d_valid / N)

                # Evaluate on training set, this time in evaluation mode
                epoch_loss_3d_train_eval = 0
                epoch_loss_traj_train_eval = 0
                epoch_loss_2d_train_labeled_eval = 0
                N = 0
                for cam, batch, batch_2d in train_generator_eval.next_epoch():
                    if batch_2d.shape[1] == 0:
                        # This can only happen when downsampling the dataset
                        continue

                    inputs_3d = torch.from_numpy(batch.astype('float32'))
                    inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                    inputs_2d, inputs_3d = eval_data_prepare(receptive_field, inputs_2d, inputs_3d)


                    if torch.cuda.is_available():
                        inputs_3d = inputs_3d.cuda(non_blocking=True)
                        inputs_2d = inputs_2d.cuda(non_blocking=True)

                    inputs_3d[:, :, 0] = 0

                    # Compute 3D poses
                    # predicted_3d_pos = model_pos(inputs_2d)

                    flag = True
                    while flag:
                        SUBJECT = S_list[random.randint(0, len(S_list) - 1)]
                        ACTION = Action_list[random.randint(0, len(Action_list) - 1)]
                        poses_act, poses_2d_act = fetch_actions(all_actions_by_subject_train[SUBJECT][ACTION])
                        if poses_act[0].shape[0] < 999:
                            continue
                        else:
                            flag = False
                        # ref_3D = torch.from_numpy(poses_act[0][:999]).float().unsqueeze(0).repeat(8, 1, 1, 1)
                        # print(ref_3D.shape)
                        # ref_3D[:, :, 0] = 0.0
                        if args.model_type == 'shift-group-dual-short':
                            start = random.randint(0, poses_act[0].shape[0] - receptive_field)
                            ref_2D = torch.from_numpy(poses_2d_act[0][start:start + receptive_field]).float().unsqueeze(
                                0).repeat(inputs_2d.shape[0], 1, 1, 1)
                        else:
                            ref_2D = torch.from_numpy(poses_2d_act[0][:999]).float().unsqueeze(0).repeat(8, 1, 1, 1)

                        if torch.cuda.is_available():
                            # ref_3D = ref_3D.cuda(non_blocking=True)
                            ref_2D = ref_2D.cuda(non_blocking=True)

                    if inputs_2d.shape[0] <= 2000:
                        if args.model_type == 'mutual' or args.model_type == 'shift-group-dual':
                            predicted_3d_pos = model_pos(ref_2D, inputs_2d)
                        elif args.model_type == 'shift-group-dual-short':
                            predicted_3d_pos = model_pos(ref_2D, inputs_2d, eval=True)
                        else:
                            predicted_3d_pos = model_pos(inputs_2d)
                        if args.bone_norm:
                            bone_length = torch.sqrt(
                                torch.sum((inputs_3d[:, :, 12:13] - inputs_3d[:, :, 13:14]) ** 2, dim=-1, keepdim=True)).mean(1, keepdim=True)
                            predicted_3d_pos = predicted_3d_pos * bone_length
                    else:
                        predicted_3d_pos = []
                        if args.model_type == 'mutual' or args.model_type == 'shift-group-dual':
                            for kk in range(inputs_2d.shape[0] // 2000 + 1):
                                predicted_3d_pos.append(
                                    model_pos(ref_2D, inputs_2d[kk * 2000: min(kk * 2000 + 2000, inputs_2d.shape[0]), ...]))
                        elif args.model_type == 'shift-group-dual-short':
                            for kk in range(inputs_2d.shape[0] // 2000 + 1):
                                predicted_3d_pos.append(
                                    model_pos(ref_2D[kk * 2000: min(kk * 2000 + 2000, inputs_2d.shape[0]), ...], inputs_2d[kk * 2000: min(kk * 2000 + 2000, inputs_2d.shape[0]), ...], eval=True))
                        else:
                            for kk in range(inputs_2d.shape[0] // 2000 + 1):
                                predicted_3d_pos.append(model_pos(inputs_2d[kk * 2000: min(kk * 2000 + 2000, inputs_2d.shape[0]), ...]))
                        predicted_3d_pos = torch.cat(predicted_3d_pos, dim=0)
                        if args.bone_norm:
                            bone_length = torch.sqrt(
                                torch.sum((inputs_3d[:, :, 12:13] - inputs_3d[:, :, 13:14]) ** 2, dim=-1, keepdim=True)).mean(1, keepdim=True)
                            predicted_3d_pos = predicted_3d_pos * bone_length


                    # del inputs_2d
                    # torch.cuda.empty_cache()

                    loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                    epoch_loss_3d_train_eval += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
                    N += inputs_3d.shape[0] * inputs_3d.shape[1]

                    # del inputs_3d, loss_3d_pos, predicted_3d_pos
                    # torch.cuda.empty_cache()

                losses_3d_train_eval.append(epoch_loss_3d_train_eval / N)

                # Evaluate 2D loss on unlabeled training set (in evaluation mode)
                epoch_loss_2d_train_unlabeled_eval = 0
                N_semi = 0

        elapsed = (time() - start_time) / 60

        if args.no_eval:
            print('[%d] time %.2f lr %f 3d_train %f' % (
                epoch + 1,
                elapsed,
                lr,
                losses_3d_train[-1] * 1000))
            writer.add_scalar('train_error', losses_3d_train[-1] * 1000, global_step=epoch)
        else:

            print('[%d] time %.2f lr %f 3d_train %f 3d_eval %f 3d_valid %f' % (
                epoch + 1,
                elapsed,
                lr,
                losses_3d_train[-1] * 1000,
                losses_3d_train_eval[-1] * 1000,
                losses_3d_valid[-1] * 1000))
            writer.add_scalar('train_error', losses_3d_train[-1] * 1000, global_step=epoch)
            writer.add_scalar('train_eval_error', losses_3d_train_eval[-1] * 1000, global_step=epoch)
            writer.add_scalar('val_error', losses_3d_valid[-1] * 1000, global_step=epoch)

        # Decay learning rate exponentially
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
        epoch += 1

        # Decay BatchNorm momentum
        # momentum = initial_momentum * np.exp(-epoch/args.epochs * np.log(initial_momentum/final_momentum))
        # model_pos_train.set_bn_momentum(momentum)

        # Save checkpoint if necessary
        if epoch % args.checkpoint_frequency == 0:
            chk_path = os.path.join(args.checkpoint, 'epoch_{}.bin'.format(epoch))
            print('Saving checkpoint to', chk_path)
            if args.torch_loader:
                rs = np.random.RandomState(1234)
            else:
                rs = train_generator.random_state()
            torch.save({
                'epoch': epoch,
                'lr': lr,
                'random_state': rs,
                'optimizer': optimizer.state_dict(),
                'model_pos': model_pos_train.state_dict(),
                # 'model_traj': model_traj_train.state_dict() if semi_supervised else None,
                # 'random_state_semi': semi_generator.random_state() if semi_supervised else None,
            }, chk_path)

        #### save best checkpoint
        best_chk_path = os.path.join(args.checkpoint, 'best_epoch.bin'.format(epoch))
        if losses_3d_valid[-1] * 1000 < min_loss:
            min_loss = losses_3d_valid[-1] * 1000
            print("save best checkpoint")
            if args.torch_loader:
                rs = np.random.RandomState(1234)
            else:
                rs = train_generator.random_state()
            torch.save({
                'epoch': epoch,
                'lr': lr,
                'random_state': rs,
                'optimizer': optimizer.state_dict(),
                'model_pos': model_pos_train.state_dict(),
                # 'model_traj': model_traj_train.state_dict() if semi_supervised else None,
                # 'random_state_semi': semi_generator.random_state() if semi_supervised else None,
            }, best_chk_path)

            print('Evaluating...')
            all_actions, all_actions_by_subject = prepare_actions()

            if not args.by_subject:
                run_evaluation(all_actions, action_filter, out=False)
            else:
                for subject in all_actions_by_subject.keys():
                    print('Evaluating on subject', subject)
                    run_evaluation(all_actions_by_subject[subject], action_filter, out=False)
                    print('')

        # Save training curves after every epoch, as .png images (if requested)
        if args.export_training_curves and epoch > 3:
            if 'matplotlib' not in sys.modules:
                import matplotlib

                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

            plt.figure()
            epoch_x = np.arange(3, len(losses_3d_train)) + 1
            plt.plot(epoch_x, losses_3d_train[3:], '--', color='C0')
            plt.plot(epoch_x, losses_3d_train_eval[3:], color='C0')
            plt.plot(epoch_x, losses_3d_valid[3:], color='C1')
            plt.legend(['3d train', '3d train (eval)', '3d valid (eval)'])
            plt.ylabel('MPJPE (m)')
            plt.xlabel('Epoch')
            plt.xlim((3, epoch))
            plt.savefig(os.path.join(args.checkpoint, 'loss_3d.png'))

            plt.close('all')



if args.render:
    print('Rendering...')

    input_keypoints = keypoints[args.viz_subject][args.viz_action][args.viz_camera].copy()
    ground_truth = None
    if args.viz_subject in dataset.subjects() and args.viz_action in dataset[args.viz_subject]:
        if 'positions_3d' in dataset[args.viz_subject][args.viz_action]:
            ground_truth = dataset[args.viz_subject][args.viz_action]['positions_3d'][args.viz_camera].copy()
    if ground_truth is None:
        print('INFO: this action is unlabeled. Ground truth will not be rendered.')

    gen = UnchunkedGenerator(None, [ground_truth], [input_keypoints],
                             pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                             kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    prediction = evaluate(gen, return_predictions=True)
    # if model_traj is not None and ground_truth is None:
    #     prediction_traj = evaluate(gen, return_predictions=True, use_trajectory_model=True)
    #     prediction += prediction_traj

    if args.viz_export is not None:
        print('Exporting joint positions to', args.viz_export)
        # Predictions are in camera space
        np.save(args.viz_export, prediction)

    if args.viz_output is not None:
        if ground_truth is not None:
            # Reapply trajectory
            trajectory = ground_truth[:, :1]
            ground_truth[:, 1:] += trajectory
            prediction += trajectory

        # Invert camera transformation
        cam = dataset.cameras()[args.viz_subject][args.viz_camera]
        if ground_truth is not None:
            prediction = camera_to_world(prediction, R=cam['orientation'], t=cam['translation'])
            ground_truth = camera_to_world(ground_truth, R=cam['orientation'], t=cam['translation'])
        else:
            # If the ground truth is not available, take the camera extrinsic params from a random subject.
            # They are almost the same, and anyway, we only need this for visualization purposes.
            for subject in dataset.cameras():
                if 'orientation' in dataset.cameras()[subject][args.viz_camera]:
                    rot = dataset.cameras()[subject][args.viz_camera]['orientation']
                    break
            prediction = camera_to_world(prediction, R=rot, t=0)
            # We don't have the trajectory, but at least we can rebase the height
            prediction[:, :, 2] -= np.min(prediction[:, :, 2])

        anim_output = {'Reconstruction': prediction}
        if ground_truth is not None and not args.viz_no_ground_truth:
            anim_output['Ground truth'] = ground_truth

        input_keypoints = image_coordinates(input_keypoints[..., :2], w=cam['res_w'], h=cam['res_h'])

        from common.visualization import render_animation

        render_animation(input_keypoints, keypoints_metadata, anim_output,
                         dataset.skeleton(), dataset.fps(), args.viz_bitrate, cam['azimuth'], args.viz_output,
                         limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
                         input_video_path=args.viz_video, viewport=(cam['res_w'], cam['res_h']),
                         input_video_skip=args.viz_skip)

else:
    print('Evaluating...')
    all_actions, all_actions_by_subject = prepare_actions()

    if not args.by_subject:
        run_evaluation(all_actions, action_filter, out=True)
    else:
        for subject in all_actions_by_subject.keys():
            print('Evaluating on subject', subject)
            run_evaluation(all_actions_by_subject[subject], action_filter, out=True)
            print('')