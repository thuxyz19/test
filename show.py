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

from common.model_ablation import *

from common.loss import *
from common.generators import ChunkedGenerator, UnchunkedGenerator, ChunkedDataset
from time import time
from common.utils import *
from tensorboardX import SummaryWriter
import shutil
from torch.utils.data import DataLoader
import random
from timm.scheduler.cosine_lr import CosineLRScheduler
import matplotlib.pyplot as plt
import pylab as mpl
from mpl_toolkits.mplot3d import Axes3D
import cv2

args = parse_args()
try:
    # Create checkpoint directory if it does not exist
    os.makedirs(args.checkpoint)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

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

model_pos = ExtSlight(num_frame=receptive_field, num_joints=num_joints, in_chans=2,
                                      embed_dim_ratio=args.embed_ratio, depth=4,
                                      num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                                      drop_path_rate=0.0, M=9, shift=3, seed=None, drop_rate=args.dropout)

causal_shift = 0

if torch.cuda.is_available():
    model_pos = nn.DataParallel(model_pos)
    model_pos = model_pos.cuda()

if args.resume or args.evaluate:
    chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    model_pos.load_state_dict(checkpoint['model_pos'], strict=False)

def eval_data_prepare(receptive_field, inputs_2d, inputs_3d):
    inputs_2d_p = torch.squeeze(inputs_2d)
    inputs_3d_p = inputs_3d.permute(1,0,2,3)
    out_num = inputs_2d_p.shape[0] - receptive_field + 1
    eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
    for i in range(out_num):
        eval_input_2d[i,:,:,:] = inputs_2d_p[i:i+receptive_field, :, :]
    return eval_input_2d, inputs_3d_p

def evaluate_batch(test_generator, action=None, return_predictions=False, use_trajectory_model=False, out=False):
    epoch_loss_3d_pos = 0
    epoch_loss_3d_pos_procrustes = 0
    epoch_loss_3d_pos_scale = 0
    epoch_loss_3d_vel = 0
    with torch.no_grad():
        if not use_trajectory_model:
            model_pos.eval()
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


            if inputs_2d.shape[0] <= 2000:
                predicted_3d_pos, _, _ = model_pos(inputs_2d)
                predicted_3d_pos_flip, _, _ = model_pos(inputs_2d_flip)
            else:
                predicted_3d_pos = []
                predicted_3d_pos_flip = []
                for kk in range(inputs_2d.shape[0] // 2000 + 1):
                    predicted_3d_pos.append(
                        model_pos(inputs_2d[kk * 2000: min(kk * 2000 + 2000, inputs_2d.shape[0]), ...])[0])
                    predicted_3d_pos_flip.append(
                        model_pos(inputs_2d_flip[kk * 2000: min(kk * 2000 + 2000, inputs_2d.shape[0]), ...])[0])
                predicted_3d_pos = torch.cat(predicted_3d_pos, dim=0)
                predicted_3d_pos_flip = torch.cat(predicted_3d_pos_flip, dim=0)

            predicted_3d_pos_flip[:, :, :, 0] *= -1
            predicted_3d_pos_flip[:, :, joints_left + joints_right] = predicted_3d_pos_flip[:, :,
                                                                      joints_right + joints_left]

            predicted_3d_pos = torch.mean(torch.cat((predicted_3d_pos, predicted_3d_pos_flip), dim=1), dim=1,
                                          keepdim=True)

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

def inference(action):
    Attn = []
    Ext = []
    poses_act, poses_2d_act = fetch_actions(action)
    gen = UnchunkedGenerator(None, poses_act, poses_2d_act,
                                pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                                kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                joints_right=joints_right)
    model_pos.eval()
    with torch.no_grad():
        for _, batch, batch_2d in gen.next_epoch():

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

            if inputs_2d.shape[0] <= 2000:
                predicted_3d_pos, attn, ext = model_pos(inputs_2d)
                predicted_3d_pos_flip, _, _ = model_pos(inputs_2d_flip)
                Attn.append(attn)
                Ext.append(ext)
            else:
                predicted_3d_pos = []
                predicted_3d_pos_flip = []
                for kk in range(inputs_2d.shape[0] // 2000 + 1):
                    predicted_3d_pos_tmp, attn_tmp, ext_tmp = model_pos(inputs_2d[kk * 2000: min(kk * 2000 + 2000, inputs_2d.shape[0]), ...])
                    predicted_3d_pos_flip_tmp, _, _ = model_pos(inputs_2d_flip[kk * 2000: min(kk * 2000 + 2000, inputs_2d.shape[0]), ...])
                    predicted_3d_pos.append(predicted_3d_pos_tmp)
                    predicted_3d_pos_flip.append(predicted_3d_pos_flip_tmp)
                    Attn.append(attn_tmp)
                    Ext.append(ext_tmp)

                predicted_3d_pos = torch.cat(predicted_3d_pos, dim=0)
                predicted_3d_pos_flip = torch.cat(predicted_3d_pos_flip, dim=0)

            predicted_3d_pos_flip[:, :, :, 0] *= -1
            predicted_3d_pos_flip[:, :, joints_left + joints_right] = predicted_3d_pos_flip[:, :,
                                                                      joints_right + joints_left]

            predicted_3d_pos = torch.mean(torch.cat((predicted_3d_pos, predicted_3d_pos_flip), dim=1), dim=1,
                                          keepdim=True)
        Attn = torch.cat(Attn, 1)
        Ext = torch.cat(Ext, 1)

    return predicted_3d_pos.squeeze().cpu().numpy(), Attn.squeeze().cpu().numpy(), Ext.squeeze().cpu().numpy()

skeleton = dataset.skeleton()
parents = skeleton.parents()
cam = dataset.cameras()
azim = 0
radius = 1.7

def draw(pose, out='./show'):
    print(pose.shape)
    # pose: (F, 17, 3) numpy array   
    fig = plt.figure()
    for i in range(pose.shape[0]):
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.view_init(elev=90., azim=azim)
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([-radius / 2, radius / 2])
        #ax.set_aspect('equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 7.5
        ax.set_title('3D人体骨架')  # , pad=35
        pos = pose[i]
        # pos = pos[:, (2, 1, 0)]
        pos = pos[:, (1, 0, 2)]
        pos[:, 2] = -1.0 * pos[:, 2]
        for j, j_parent in enumerate(parents):
            if j_parent == -1:
                continue
            col = 'red' if j in skeleton.joints_right() else 'black'
            ax.plot([pos[j, 0], pos[j_parent, 0]],[pos[j, 1], pos[j_parent, 1]],[pos[j, 2], pos[j_parent, 2]], zdir='z', c=col)
        plt.savefig(os.path.join(out,'%06d.jpg'%i))
        fig.clear()

print('Evaluating...')
all_actions, all_actions_by_subject = prepare_actions()
run_evaluation(all_actions, action_filter, out=True)
# print(all_actions.keys())
action =(('S9', 'Directions'),)
pose, self_attn, ext_attn = inference(action)
print(self_attn.shape, ext_attn.shape)
# draw(pose)
# for i in range(5):
#     attn_path = os.path.join('./attn', str(i))
#     for j in range(attn[i].shape[0]):
#         attn[i, j] = np.exp(attn[i, j] * 1024.0)
#         attn[i, j] = (attn[i, j] - np.min(attn[i, j])) / (np.max(attn[i, j]) - np.min(attn[i, j]))
#         cv2.imwrite(os.path.join(attn_path, str(j)+'.jpg'), (attn[i, j] * 255).astype(np.uint8))
# seed = model_pos.module.seed.cpu().numpy()
# draw(seed, out='./seed')
