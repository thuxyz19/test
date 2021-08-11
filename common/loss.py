# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np
from torch._C import device


parts = ((1, 2, 3), (4, 5, 6), (0, 7, 8, 9, 10), (11, 12, 13), (14, 15, 16))

def part_mpjpe(predicted, target, valid=None):
    e = []
    for p in parts:
        pred_tmp = predicted[:, p, :]
        targ_tmp = target[:, p, :]
        pred_tmp = pred_tmp[:, 1:, :] - pred_tmp[:, 0:1, :]
        targ_tmp = targ_tmp[:, 1:, :] - targ_tmp[:, 0:1, :]
        p_e = mpjpe(pred_tmp, targ_tmp)
        e.append(p_e)
    return e[0].item(), e[1].item(), e[2].item(), e[3].item(), e[4].item()

def part_angle(predicted, target, valid=None):
    e = []
    for p in parts:
        pred_tmp = predicted[:, :, p, :]
        targ_tmp = target[:, :, p, :]
        pred_angle = torch.sum((pred_tmp[:, :, 0, :] - pred_tmp[:, :, 1, :]) * (pred_tmp[:, :, 2, :] - pred_tmp[:, :, 1, :]), -1)
        targ_angle = torch.sum(
            (targ_tmp[:, :, 0, :] - targ_tmp[:, :, 1, :]) * (targ_tmp[:, :, 2, :] - targ_tmp[:, :, 1, :]), -1)
        pred_angle = pred_angle / (torch.norm(pred_tmp[:, :, 0, :] - pred_tmp[:, :, 1, :], -1) * torch.norm(pred_tmp[:, :, 2, :] - pred_tmp[:, :, 1, :], -1))
        targ_angle = targ_angle / (torch.norm(targ_tmp[:, :, 0, :] - targ_tmp[:, :, 1, :], -1) * torch.norm(
            targ_tmp[:, :, 2, :] - targ_tmp[:, :, 1, :], -1))
        pred_angle = torch.acos(pred_angle)
        targ_angle = torch.acos(targ_angle)
        p_e = (torch.abs(pred_angle - targ_angle)).mean()
        e.append(p_e)
    return e



def euclidean_losses(actual, target):
    """Calculate the average Euclidean loss for multi-point samples.
    Each sample must contain `n` points, each with `d` dimensions. For example,
    in the MPII human pose estimation task n=16 (16 joint locations) and
    d=2 (locations are 2D).
    Args:
        actual (Tensor): Predictions (B x L x D)
        target (Tensor): Ground truth target (B x L x D)
    """

    assert actual.size() == target.size(), 'input tensors must have the same size'

    # Calculate Euclidean distances between actual and target locations
    diff = actual - target
    dist_sq = diff.pow(2).sum(-1, keepdim=False)
    dist = dist_sq.sqrt()
    return dist


def pck(actual, expected, included_joints=None, threshold=0.15, valid=None):
    dists = euclidean_losses(actual, expected)
    if included_joints is not None:
        dists = dists.gather(-1, torch.LongTensor(included_joints))
    if valid is not None:
        valid = valid.view(dists.shape[0], 1, 1)
    else:
        valid = torch.ones((dists.shape[0], 1, 1), dtype=torch.float32, device=dists.device)

    return ((dists < threshold).double()*valid).mean().item() * dists.shape[0] / valid.sum()


def auc(actual, expected, included_joints=None, valid=None):
    # This range of thresholds mimics `mpii_compute_3d_pck.m`, which is provided as part of the
    # MPI-INF-3DHP test data release.
    thresholds = torch.linspace(0, 150, 31).tolist()

    pck_values = torch.DoubleTensor(len(thresholds))
    for i, threshold in enumerate(thresholds):
        pck_values[i] = pck(actual, expected, included_joints, threshold=threshold/1000.0, valid=valid)
    return pck_values.mean().item()


def motion_loss(predicted, target):
    '''
    The motion loss
    '''
    predicted = predicted.permute(0, 2, 1, 3).reshape(-1, 9, 3)
    target = target.permute(0, 2, 1, 3).reshape(-1, 9, 3)
    predicted_a = predicted.repeat(1, 9, 1)
    predicted_b = predicted.view(-1, 9, 1, 3).repeat(1, 1, 9, 1).view(-1, 81, 3)
    predicted_motion = torch.cross(predicted_a, predicted_b)
    target_a = target.repeat(1, 9, 1)
    target_b = target.view(-1, 9, 1, 3).repeat(1, 1, 9, 1).view(-1, 81, 3)
    target_motion = torch.cross(target_a, target_b)
    return torch.mean(torch.norm(predicted_motion - target_motion, dim=len(target_motion.shape) - 1))



def mpjpe(predicted, target, valid=None):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    if valid is not None:
        valid = valid.view(valid.shape[0], 1, 1)
    else:
        valid = torch.ones((target.shape[0], 1, 1), dtype=torch.float32, device=target.device)
    err = torch.norm(predicted - target, dim=len(target.shape)-1)
    err = err * valid
    e = err.mean() * err.shape[0] / valid.sum()
    return e
    
def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape)-1))

def p_mpjpe(predicted, target, valid=None):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape
    
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    
    # Return MPJPE
    if valid is not None:
        valid = valid.view(valid.shape[0], 1, 1).cpu().numpy()
    else:
        valid = np.ones((target.shape[0], 1, 1), dtype=np.float32)
    err = np.linalg.norm(predicted_aligned - target,  axis=len(target.shape)-1) * valid

    return np.mean(err) * err.shape[0] / np.sum(valid)
    
def n_mpjpe(predicted, target, valid=None):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape
    
    norm_predicted = torch.mean(torch.sum(predicted**2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target*predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return mpjpe(scale * predicted, target, valid)

def weighted_bonelen_loss(predict_3d_length, gt_3d_length):
    loss_length = 0.001 * torch.pow(predict_3d_length - gt_3d_length, 2).mean()
    return loss_length

def weighted_boneratio_loss(predict_3d_length, gt_3d_length):
    loss_length = 0.1 * torch.pow((predict_3d_length - gt_3d_length)/gt_3d_length, 2).mean()
    return loss_length

def mean_velocity_error(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    
    velocity_predicted = np.diff(predicted, axis=0)
    velocity_target = np.diff(target, axis=0)
    
    return np.mean(np.linalg.norm(velocity_predicted - velocity_target, axis=len(target.shape)-1))
