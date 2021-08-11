#import sys
#sys.path.append('/home/xueyz/mmdetection/mmdet/ops/')
#from dcn.deform_conv import DeformConv
from copy import deepcopy
import numpy as np
import pickle
import random

from scipy.optimize import least_squares

import torch
from torch import nn

from mvn_new.utils import op, multiview, img, misc, volumetric
from mvn_new.models.TemporalNet import TemporalNet
from mvn_new.models.transformer import MotionNet
from mvn_new.models import pose_resnet, pose_hrnet, st_gcn_heatmap, ParallelTransformer
from mvn_new.models.stn import STN
from mvn_new.models.v2v import V2VModel, V2VModel_point
from mvn_new.models.depth import DepthNet, get_init_depth, create_idx
import torch.nn.functional as Fu
import random
from SMPL import batch_transform
import matplotlib.pyplot as plt
from mvn_new.utils.img import denormalize_image
from mvn_new.utils.opt1 import opts
import time
import os
parents= [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]

class PointNet(nn.Module):
    def __init__(self,input_points,input_features, joints,cuboid_side):
        super().__init__()
        self.point_num = input_points
        self.features = input_features
        self.joints = joints
        self.cuboid_side = cuboid_side
        #self.coordinate = nn.Sequential(nn.Linear(3,64),nn.ReLU(True),nn.Linear(64,128),nn.ReLU(True))
        #self.att = nn.Sequential(nn.Linear(self.features,128),nn.ReLU(True),nn.Linear(128,1),nn.Sigmoid())
        self.global_feature = nn.Sequential(nn.Linear(self.features,64),nn.ReLU(True),nn.Linear(64,32),nn.ReLU(True))
        self.fusion = nn.Sequential(nn.Linear(self.features+32,64), nn.ReLU(True),nn.Linear(64,64),nn.ReLU(True))
        self.output = nn.Sequential(nn.Linear(64,64),nn.ReLU(True),nn.Linear(64,64),nn.ReLU(True),nn.Linear(64,joints))
    def forward(self,x): #x:(B,features+4,64,64,64) features+xyz+q
        B = x.shape[0]
        x = x.view(B,self.features+4,-1)
        x = x.permute(0,2,1)
        _,indices = x.sort(1)
        pc = []
        for b in range(B):
            pc.append(x[b,indices[b,-self.point_num:,-1],:-1]) #(k,features+3)
        pc = torch.stack(pc,0) #(B,k,features+3)
        pc = pc.view(-1,self.features+3)
        coord = pc[:,-3:].view(B,-1,3)
        features = pc[:,:-3]
        features_global = self.global_feature(features)
        features_global = features_global.view(B, -1, 32).mean(1).view(B,1,32).repeat(1,self.point_num,1).view(-1,32)
        features = torch.cat([features,features_global],1)
        features = self.output(self.fusion(features)) #(-1,17)
        features = features.view(B,-1,17)
        weights = Fu.softmax(features,1) #(B,-1,17)
        pred = torch.stack([(weights*coord[:,:,0].view(B,-1,1).repeat(1,1,17)).sum(1),(weights*coord[:,:,1].view(B,-1,1).repeat(1,1,17)).sum(1),(weights*coord[:,:,2].view(B,-1,1).repeat(1,1,17)).sum(1)],-1)
        return pred

class RANSACTriangulationNet(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()

        config.model.backbone.alg_confidences = False
        config.model.backbone.vol_confidences = False
        self.backbone = pose_resnet.get_pose_net(config.model.backbone)
        # self.return_confidences = config.backbone.return_confidences
        self.direct_optimization = config.model.direct_optimization

    def forward(self, images, proj_matricies, batch):
        batch_size, n_views = images.shape[:2]

        # reshape n_views dimension to batch dimension
        images = images.view(-1, *images.shape[2:])

        # forward backbone and integrate
        heatmaps, _, _, _ = self.backbone(images)

        # reshape back
        images = images.view(batch_size, n_views, *images.shape[1:])
        heatmaps = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])

        # calcualte shapes
        image_shape = tuple(images.shape[3:])
        batch_size, n_views, n_joints, heatmap_shape = heatmaps.shape[0], heatmaps.shape[1], heatmaps.shape[2], tuple(heatmaps.shape[3:])

        # keypoints 2d
        _, max_indicies = torch.max(heatmaps.view(batch_size, n_views, n_joints, -1), dim=-1)
        keypoints_2d = torch.stack([max_indicies % heatmap_shape[1], max_indicies // heatmap_shape[1]], dim=-1).to(images.device)

        # upscale keypoints_2d, because image shape != heatmap shape
        keypoints_2d_transformed = torch.zeros_like(keypoints_2d)
        keypoints_2d_transformed[:, :, :, 0] = keypoints_2d[:, :, :, 0] * (image_shape[1] / heatmap_shape[1])
        keypoints_2d_transformed[:, :, :, 1] = keypoints_2d[:, :, :, 1] * (image_shape[0] / heatmap_shape[0])
        keypoints_2d = keypoints_2d_transformed

        # triangulate (cpu)
        keypoints_2d_np = keypoints_2d.detach().cpu().numpy()
        proj_matricies_np = proj_matricies.detach().cpu().numpy()

        keypoints_3d = np.zeros((batch_size, n_joints, 3))
        confidences = np.zeros((batch_size, n_views, n_joints))  # plug
        for batch_i in range(batch_size):
            for joint_i in range(n_joints):
                current_proj_matricies = proj_matricies_np[batch_i]
                points = keypoints_2d_np[batch_i, :, joint_i]
                keypoint_3d, _ = self.triangulate_ransac(current_proj_matricies, points, direct_optimization=self.direct_optimization)
                keypoints_3d[batch_i, joint_i] = keypoint_3d

        keypoints_3d = torch.from_numpy(keypoints_3d).type(torch.float).to(images.device)
        confidences = torch.from_numpy(confidences).type(torch.float).to(images.device)

        return keypoints_3d, keypoints_2d, heatmaps, confidences

    def triangulate_ransac(self, proj_matricies, points, n_iters=10, reprojection_error_epsilon=15, direct_optimization=True):
        assert len(proj_matricies) == len(points)
        assert len(points) >= 2

        proj_matricies = np.array(proj_matricies)
        points = np.array(points)

        n_views = len(points)

        # determine inliers
        view_set = set(range(n_views))
        inlier_set = set()
        for i in range(n_iters):
            sampled_views = sorted(random.sample(view_set, 2))

            keypoint_3d_in_base_camera = multiview.triangulate_point_from_multiple_views_linear(proj_matricies[sampled_views], points[sampled_views])
            reprojection_error_vector = multiview.calc_reprojection_error_matrix(np.array([keypoint_3d_in_base_camera]), points, proj_matricies)[0]

            new_inlier_set = set(sampled_views)
            for view in view_set:
                current_reprojection_error = reprojection_error_vector[view]
                if current_reprojection_error < reprojection_error_epsilon:
                    new_inlier_set.add(view)

            if len(new_inlier_set) > len(inlier_set):
                inlier_set = new_inlier_set

        # triangulate using inlier_set
        if len(inlier_set) == 0:
            inlier_set = view_set.copy()

        inlier_list = np.array(sorted(inlier_set))
        inlier_proj_matricies = proj_matricies[inlier_list]
        inlier_points = points[inlier_list]

        keypoint_3d_in_base_camera = multiview.triangulate_point_from_multiple_views_linear(inlier_proj_matricies, inlier_points)
        reprojection_error_vector = multiview.calc_reprojection_error_matrix(np.array([keypoint_3d_in_base_camera]), inlier_points, inlier_proj_matricies)[0]
        reprojection_error_mean = np.mean(reprojection_error_vector)

        keypoint_3d_in_base_camera_before_direct_optimization = keypoint_3d_in_base_camera
        reprojection_error_before_direct_optimization = reprojection_error_mean

        # direct reprojection error minimization
        if direct_optimization:
            def residual_function(x):
                reprojection_error_vector = multiview.calc_reprojection_error_matrix(np.array([x]), inlier_points, inlier_proj_matricies)[0]
                residuals = reprojection_error_vector
                return residuals

            x_0 = np.array(keypoint_3d_in_base_camera)
            res = least_squares(residual_function, x_0, loss='huber', method='trf')

            keypoint_3d_in_base_camera = res.x
            reprojection_error_vector = multiview.calc_reprojection_error_matrix(np.array([keypoint_3d_in_base_camera]), inlier_points, inlier_proj_matricies)[0]
            reprojection_error_mean = np.mean(reprojection_error_vector)

        return keypoint_3d_in_base_camera, inlier_list


def update_after_resize(K, image_shape, new_image_shape):
    height, width = image_shape
    new_width, new_height = new_image_shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    new_K = torch.zeros_like(K)
    new_K[0, 0] = fx * (new_width / width)
    new_K[1, 1] = fy * (new_height / height)
    new_K[0, 2] = cx * (new_width / width)
    new_K[1, 2] = cy * (new_height / height)

    return new_K


class TransformerNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.backbone_type = config.model.backbone.name
        self.channel = 16
        if self.backbone_type == 'resnet152':
            self.conv = nn.Sequential(nn.Conv2d(256, self.channel, kernel_size=(1, 1)), nn.BatchNorm2d(self.channel), nn.ReLU())
        else:
            self.conv = nn.Sequential(nn.Conv2d(32, self.channel, kernel_size=(1, 1)), nn.BatchNorm2d(self.channel), nn.ReLU())

        config.model.backbone.alg_confidences = True
        config.model.backbone.vol_confidences = None
        if config.model.backbone.name == 'hrnet' and config.image_shape[0] == 256:
            self.backbone = pose_hrnet.get_pose_net(config, is_train=True, mask=config.mask)
            state_dict = torch.load(config.TEST.MODEL_FILE)
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[k.replace('backbone.', '')] = v
            self.backbone.load_state_dict(new_state_dict, strict=False)
        else:
            self.backbone = pose_resnet.get_pose_net(config.model.backbone)

        self.heatmap_softmax = config.model.heatmap_softmax
        self.heatmap_multiplier = config.model.heatmap_multiplier
        self.sigma = config.model.sigma
        self.pad_len = 8

        self.transformer_net = ParallelTransformer.Transformer(in_channels=self.channel + 2, use_confidences=config.model.backbone.alg_confidences)


    def forward(self, images, proj_matricies, frames=8):
        # print(images.shape)  # [batch*frames, 4, 3, 384, 384]
        # keypoints_3d_history [batch, frames-1, 17, 3]
        batch_size, n_views = images.shape[:2]
        # forward backbone and integral
        # reshape n_views dimension to batch dimension
        images = images.view(-1, *images.shape[2:])  # 64x3x384x384
        image_shape = tuple(images.shape[-2:])
        heatmaps, features, alg_confidences_history = self.backbone(images)
        heatmap_shape = tuple(heatmaps.shape[-2:])

        if self.backbone_type == 'resnet152':
            heatmaps = heatmaps[:, (6, 2, 1, 0, 3, 4, 5, 7, 8, 16, 9, 13, 14, 15, 12, 11, 10), :, :]
            alg_confidences_history = alg_confidences_history[:, (6, 2, 1, 0, 3, 4, 5, 7, 8, 16, 9, 13, 14, 15, 12, 11, 10)]

        if alg_confidences_history is not None:
            alg_confidences_history = alg_confidences_history.view(batch_size, n_views, *alg_confidences_history.shape[1:])
            alg_confidences_history = alg_confidences_history / torch.clamp_min(alg_confidences_history.sum(dim=1, keepdim=True), min=1e-5)
            alg_confidences_history = alg_confidences_history + 1e-5  # for numerical stability



        keypoints_2d_origin, _ = op.integrate_tensor_2d(heatmaps * self.heatmap_multiplier, self.heatmap_softmax)  # 64x17x2

        features = self.conv(features)
        # print(features.shape)  # 64x16x96x96
        features = features.view(batch_size, n_views, *features.shape[1:]).transpose(1, 2).contiguous().transpose(0,
                                                                                                                      1).contiguous()
        features = features.view(-1, *features.shape[2:])
        features = features.unsqueeze(2).repeat(1, 1, 17, 1, 1)  # 16B x n_views x 17 x H x W
        keypoints_2d = keypoints_2d_origin.view(batch_size // frames, frames, n_views, -1, 2)  # [B, T, n_views, 17, 2] = [2, 8, 4, 17, 2]
        position_matrix = keypoints_2d.transpose(1, 2).contiguous().cuda()  # [B, n_views, T, 17, 2] = [2, 4, 8, 17, 2]

        keypoints_2d = keypoints_2d.view(-1, *keypoints_2d.shape[2:])

        pad_len = self.pad_len
        pad_feature = Fu.pad(features, [pad_len, pad_len,pad_len, pad_len])
        keypoints_2d_x = keypoints_2d[:, :, :, 0:1].repeat(self.channel, 1, 1, 2*pad_len).long() + torch.arange(0, 2*pad_len).view(1, 1, 1, -1).cuda()
        keypoints_2d_x = keypoints_2d_x.unsqueeze(3).repeat(1, 1, 1, pad_feature.shape[-2], 1)
        cropped_feature_x = torch.gather(pad_feature, 4, keypoints_2d_x)
        keypoints_2d_y = keypoints_2d[:, :, :, 1:2].repeat(self.channel, 1, 1, 2 * pad_len).long() + torch.arange(0, 2*pad_len).view(
            1, 1, 1, -1).cuda()
        keypoints_2d_y = keypoints_2d_y.unsqueeze(4).repeat(1, 1, 1, 1, 2 * pad_len)

        cropped_feature = torch.gather(cropped_feature_x, 3, keypoints_2d_y)

        cropped_feature = cropped_feature.view(self.channel, batch_size // frames, frames, n_views,
                                                   *cropped_feature.shape[2:]).transpose(3, 2).contiguous().transpose(3, 4).contiguous()
        cropped_feature = cropped_feature.view(self.channel, -1, *cropped_feature.shape[3:])  # (C, B * n_views, 17, T, H, W) = [16, 8, 17, 8, 16, 16]

        cropped_feature = cropped_feature.view(self.channel, batch_size//frames, n_views, *cropped_feature.shape[2:]).transpose(3, 4).contiguous()
        cropped_feature = cropped_feature.permute(1, 2, 3, 4, 0, 5, 6)  # [2, 4, 8, 17, 16, 16, 16] --> [2, 4, 8, 17, 18, 16, 16] B, views, frames, J, C, H, W
        # position_matrix = position_matrix.view(position_matrix.shape[0], position_matrix.shape[1]//n_views, n_views, *position_matrix.shape[2:], ).permute(1, 2, 4, 3, 0, 5, 6)
        cropped_feature = cropped_feature.mean(-1).mean(-1)
        cropped_feature = torch.cat((cropped_feature, position_matrix), dim=4)

        keypoints_2d_refine, alg_confidences = self.transformer_net(cropped_feature)  # (B, n-views, T, 17, 2), (B, n_views, T, 17)

        alg_confidences = alg_confidences.view(batch_size // frames, n_views, frames, -1).transpose(1, 2).contiguous().view(batch_size*n_views, -1)


        # reshape back
        heatmaps = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])
        if alg_confidences is not None:
            alg_confidences = alg_confidences.view(batch_size, n_views, *alg_confidences.shape[1:])
            alg_confidences = alg_confidences / torch.clamp_min(alg_confidences.sum(dim=1, keepdim=True), min=1e-5)
            alg_confidences = alg_confidences + 1e-5  # for numerical stability

        # calcualte shapes
        batch_size, n_views, n_joints, heatmap_shape = heatmaps.shape[0], heatmaps.shape[1], heatmaps.shape[2], tuple(heatmaps.shape[3:])

        # upscale keypoints_2d, because image shape != heatmap shape
        keypoints_2d_transformed = torch.zeros_like(keypoints_2d_refine)
        keypoints_2d_transformed[:, :, :, 0] = keypoints_2d_refine[:, :, :, 0] * (image_shape[1] / heatmap_shape[1])
        keypoints_2d_transformed[:, :, :, 1] = keypoints_2d_refine[:, :, :, 1] * (image_shape[0] / heatmap_shape[0])

        keypoints_3d = multiview.triangulate_batch_of_points(
            proj_matricies, keypoints_2d_transformed,
            confidences_batch=alg_confidences
        )
        keypoints_3d = keypoints_3d * 1000.0

        keypoints_2d_new = keypoints_2d_origin.view(-1, n_views, *keypoints_2d_origin.shape[-2:])
        keypoints_2d_transformed = torch.zeros_like(keypoints_2d_new)
        keypoints_2d_transformed[:, :, :, 0] = keypoints_2d_new[:, :, :, 0] * (image_shape[1] / heatmap_shape[1])
        keypoints_2d_transformed[:, :, :, 1] = keypoints_2d_new[:, :, :, 1] * (image_shape[0] / heatmap_shape[0])

        keypoints_3d_pre = multiview.triangulate_batch_of_points(
            proj_matricies, keypoints_2d_transformed,
            confidences_batch=alg_confidences_history
        )
        keypoints_3d_pre = keypoints_3d_pre * 1000.0


        return keypoints_3d, keypoints_3d_pre, keypoints_2d_refine * 4.0, heatmaps, alg_confidences

    def generate_target(self, keypoints_2d_gt, keypoints_2d_mean, image_size, heatmap_size, cropped_heatmap_size):
        '''
        :param joints:  [batch, num_joints, 2]
        :return: target
        '''

        feat_stride = (image_size / heatmap_size).cuda()
        mu_x = (keypoints_2d_gt[:, :, 0] / feat_stride[0] + 0.5).type(torch.long) - keypoints_2d_mean[:, :, 0] + cropped_heatmap_size[1] // 2
        mu_y = (keypoints_2d_gt[:, :, 1] / feat_stride[1] + 0.5).type(torch.long) - keypoints_2d_mean[:, :, 1] + cropped_heatmap_size[0] // 2
        keypoints_validity = (mu_x < cropped_heatmap_size[1]).float() * (mu_x >= 0).float() * (mu_y < cropped_heatmap_size[0]).float() * (mu_y >= 0).float()

        tmp_size = self.sigma * 3
        target = torch.zeros((keypoints_2d_gt.shape[0], keypoints_2d_gt.shape[1],
                           cropped_heatmap_size[0] + 2 * tmp_size,
                           cropped_heatmap_size[1] + 2 * tmp_size),
                          dtype=torch.float32).cuda()
        target = target.view(*target.shape[:2], -1)
        # Generate gaussian
        size = 2 * tmp_size + 1
        x = torch.arange(start=0, end=size, step=1, out=None).reshape(1, size)
        x = x.repeat(size, 1).cuda()
        y = x.transpose(0, 1)
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = torch.exp((-((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2)).float())
        g = g.unsqueeze(0).unsqueeze(0).repeat(keypoints_2d_gt.shape[0], keypoints_2d_gt.shape[1], 1, 1).view(keypoints_2d_gt.shape[0], keypoints_2d_gt.shape[1], -1)
        index_x_before = (
                    torch.arange(start=0, end=size, step=1).cuda().reshape(1, 1, size).repeat(keypoints_2d_gt.shape[0],
                                                                                              keypoints_2d_gt.shape[1],
                                                                                              1) + mu_x.unsqueeze(
                -1).repeat(1, 1, size)).unsqueeze(-2).repeat(1, 1, size, 1)
        index_x = torch.min(torch.max(torch.tensor(0), index_x_before.cpu()), torch.tensor(cropped_heatmap_size[1] + 2 * tmp_size - 1)).cuda()
        index_y_before = (
                    torch.arange(start=0, end=size, step=1).cuda().reshape(1, 1, size).repeat(keypoints_2d_gt.shape[0],
                                                                                              keypoints_2d_gt.shape[1],
                                                                                                    1) + mu_y.unsqueeze(
                -1).repeat(1, 1, size)).unsqueeze(-1).repeat(1, 1, 1, size)
        index_y = torch.min(torch.max(torch.tensor(0), index_y_before.cpu()), torch.tensor(cropped_heatmap_size[0] + 2 * tmp_size - 1)).cuda()
        index_y *= cropped_heatmap_size[1] + 2 * tmp_size
        index = (index_x + index_y).view(keypoints_2d_gt.shape[0], keypoints_2d_gt.shape[1], -1)
        # print(target.shape, index.shape, g.shape)
        target.scatter_(-1, index, g)
        target = target.view(keypoints_2d_gt.shape[0], keypoints_2d_gt.shape[1], cropped_heatmap_size[0] + 2*tmp_size, cropped_heatmap_size[1] + 2*tmp_size)
        target = target[:, :, tmp_size: -tmp_size, tmp_size: -tmp_size]

        return target, keypoints_validity


class AlgebraicTriangulationNetViews(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.backbone_type = config.model.backbone.name
        self.mask_subsequent = config.model.mask_subsequent
        self.channel = 16
        if self.backbone_type == 'resnet152':
            self.conv = nn.Sequential(nn.Conv2d(256, self.channel, kernel_size=(1, 1)), nn.BatchNorm2d(self.channel), nn.ReLU())
        else:
            self.conv = nn.Sequential(nn.Conv2d(32, self.channel, kernel_size=(1, 1)), nn.BatchNorm2d(self.channel), nn.ReLU())

        config.model.backbone.alg_confidences = True
        config.model.backbone.vol_confidences = None
        if config.model.backbone.name == 'hrnet' and config.image_shape[0] == 256:
            self.backbone = pose_hrnet.get_pose_net(config, is_train=True, mask=config.mask)
            state_dict = torch.load(config.TEST.MODEL_FILE)
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[k.replace('backbone.', '')] = v
            self.backbone.load_state_dict(new_state_dict, strict=False)

        else:
            self.backbone = pose_resnet.get_pose_net(config.model.backbone)

        self.heatmap_softmax = config.model.heatmap_softmax
        self.heatmap_multiplier = config.model.heatmap_multiplier
        self.sigma = config.model.sigma
        self.pad_len = 8

        self.transformer_net = st_gcn_heatmap.TransformerViews(mask_subsequent=self.mask_subsequent)


    def forward(self, images, K, proj_matricies, keypoints_2d_gt, frames=8):
        # print(images.shape)  # [batch*frames, 4, 3, 384, 384]
        # keypoints_3d_history [batch, frames-1, 17, 3]
        batch_size, n_views = images.shape[:2]
        # forward backbone and integral
        # reshape n_views dimension to batch dimension
        images = images.view(-1, *images.shape[2:])  # 64x3x384x384
        image_shape = tuple(images.shape[-2:])
        heatmaps, features, alg_confidences_history = self.backbone(images)
        heatmap_shape = tuple(heatmaps.shape[-2:])

        if self.backbone_type == 'resnet152':
            heatmaps = heatmaps[:, (6, 2, 1, 0, 3, 4, 5, 7, 8, 16, 9, 13, 14, 15, 12, 11, 10), :, :]
            alg_confidences_history = alg_confidences_history[:, (6, 2, 1, 0, 3, 4, 5, 7, 8, 16, 9, 13, 14, 15, 12, 11, 10)]

        if alg_confidences_history is not None:
            alg_confidences_history = alg_confidences_history.view(batch_size, n_views, *alg_confidences_history.shape[1:])
            alg_confidences_history = alg_confidences_history / torch.clamp_min(alg_confidences_history.sum(dim=1, keepdim=True), min=1e-5)
            alg_confidences_history = alg_confidences_history + 1e-5  # for numerical stability


        heatmaps_before_softmax = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])

        keypoints_2d_origin, _ = op.integrate_tensor_2d(heatmaps * self.heatmap_multiplier, self.heatmap_softmax)  # 64x17x2

        keypoints_2d_new = keypoints_2d_origin.view(-1, n_views, *keypoints_2d_origin.shape[-2:])
        keypoints_2d_transformed = torch.zeros_like(keypoints_2d_new)
        keypoints_2d_transformed[:, :, :, 0] = keypoints_2d_new[:, :, :, 0] * (image_shape[1] / heatmap_shape[1])
        keypoints_2d_transformed[:, :, :, 1] = keypoints_2d_new[:, :, :, 1] * (image_shape[0] / heatmap_shape[0])

        keypoints_3d_pre = multiview.triangulate_batch_of_points(
            proj_matricies, keypoints_2d_transformed,
            confidences_batch=alg_confidences_history
        )
        keypoints_3d_pre = keypoints_3d_pre * 1000.0
        keypoints_3d_omits = []
        for i in range(n_views):
            if i == 0:
                idx = (1, 2, 3)
            elif i == 1:
                idx = (0, 2, 3)
            elif i == 2:
                idx = (0, 1, 3)
            else:
                idx = (0, 1, 2)
            proj_matricies_omit = proj_matricies[:, idx, :, :]
            keypoints_2d_transformed_omit = keypoints_2d_transformed[:, idx, :, :]
            alg_confidences_omit = alg_confidences_history[:, idx, :]
            # print(alg_confidences_omit.shape, keypoints_2d_transformed_omit.shape, proj_matricies_omit.shape)
            keypoints_3d_omit = multiview.triangulate_batch_of_points(
                proj_matricies_omit, keypoints_2d_transformed_omit,
                confidences_batch=alg_confidences_omit
            )
            keypoints_3d_omits.append(keypoints_3d_omit * 1000.0)
        keypoints_3d_omits = torch.stack(keypoints_3d_omits, dim=0)




        features = self.conv(features)
        # print(features.shape)  # 64x16x64x64
        features = features.view(batch_size, n_views, *features.shape[1:]).transpose(1, 2).contiguous().transpose(0,
                                                                                                                      1).contiguous()
        features = features.view(-1, *features.shape[2:])
        features = features.unsqueeze(2).repeat(1, 1, 17, 1, 1)  # 16B x n_views x 17 x H x W
        keypoints_2d = keypoints_2d_origin.view(batch_size // frames, frames, n_views, -1, 2)  # [B, T, n_views, 17, 2] = [2, 8, 4, 17, 2]
        position_matrix = keypoints_2d.transpose(1, 2).contiguous().cuda()  # [B, n_views, T, 17, 2] = [2, 4, 8, 17, 2]
        position_matrix = position_matrix.view(-1, *position_matrix.shape[2:]).transpose(1, 2).contiguous().permute(3, 0, 1, 2).contiguous()  # [2, Bxn_views, 17, T] = [2, 8, 17, 8]
        position_matrix = position_matrix.reshape(*position_matrix.shape, 1, 1).repeat(1, 1, 1, 1, 2 * self.pad_len, 1).repeat(1, 1, 1, 1, 1, 2 * self.pad_len)  # [2, Bxn_views, 17, T, 16, 16] = [2, 8, 17, 8, 16, 16]
        H_range_matrix = (torch.arange(-self.pad_len, self.pad_len).cuda() + 0.5).reshape(1, 1, 1, 2 * self.pad_len, 1).repeat(*position_matrix.shape[1:4], 1, position_matrix.shape[-1])
        W_range_matrix = (torch.arange(-self.pad_len, self.pad_len).cuda() + 0.5).reshape(1, 1, 1, 1, 2 * self.pad_len).repeat(*position_matrix.shape[1:4], position_matrix.shape[-1], 1)
        position_matrix += torch.stack([W_range_matrix, H_range_matrix], dim=0)
        keypoints_2d = keypoints_2d.mean(dim=1, keepdim=True).long().repeat(1, frames, 1, 1, 1)
        keypoints_2d = keypoints_2d.view(-1, *keypoints_2d.shape[2:])

        pad_len = self.pad_len
        pad_feature = Fu.pad(features, [pad_len, pad_len,pad_len, pad_len])
        keypoints_2d_x = keypoints_2d[:, :, :, 0:1].repeat(self.channel, 1, 1, 2*pad_len).long() + torch.arange(0, 2*pad_len).view(1, 1, 1, -1).cuda()
        keypoints_2d_x = keypoints_2d_x.unsqueeze(3).repeat(1, 1, 1, pad_feature.shape[-2], 1)
        cropped_feature_x = torch.gather(pad_feature, 4, keypoints_2d_x)
        keypoints_2d_y = keypoints_2d[:, :, :, 1:2].repeat(self.channel, 1, 1, 2 * pad_len).long() + torch.arange(0, 2*pad_len).view(
            1, 1, 1, -1).cuda()
        keypoints_2d_y = keypoints_2d_y.unsqueeze(4).repeat(1, 1, 1, 1, 2 * pad_len)

        cropped_feature = torch.gather(cropped_feature_x, 3, keypoints_2d_y)

        cropped_feature = cropped_feature.view(self.channel, batch_size // frames, frames, n_views,
                                                   *cropped_feature.shape[2:]).transpose(3, 2).contiguous().transpose(3, 4).contiguous()
        cropped_feature = cropped_feature.view(self.channel, -1, *cropped_feature.shape[3:])  # (C, B * n_views, 17, T, H, W) = [16, 8, 17, 8, 16, 16]

        cropped_feature = cropped_feature.view(self.channel, batch_size//frames, n_views, *cropped_feature.shape[2:]).transpose(3, 4).contiguous()
        cropped_feature = cropped_feature.permute(1, 2, 3, 4, 0, 5, 6)  # [2, 4, 8, 17, 16, 16, 16] --> [2, 4, 8, 17, 18, 16, 16]
        position_matrix = position_matrix.view(position_matrix.shape[0], position_matrix.shape[1]//n_views, n_views, *position_matrix.shape[2:], ).permute(1, 2, 4, 3, 0, 5, 6)

        cropped_feature = torch.cat((cropped_feature, position_matrix), dim=4)

        keypoints_3d_omits = keypoints_3d_omits.view(n_views, batch_size // frames, frames, 17, 3).transpose(1, 0).contiguous().unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 1, 1, pad_len*2, pad_len*2)
        cropped_feature = torch.cat([cropped_feature, keypoints_3d_omits], dim=4)

        alg_confidences = self.transformer_net(cropped_feature)  # (B*n-views*T, 17)

        alg_confidences = alg_confidences.view(batch_size // frames, n_views, frames, -1).transpose(1, 2).contiguous().view(batch_size*n_views, -1)


        # reshape back
        heatmaps = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])
        if alg_confidences is not None:
            alg_confidences = alg_confidences.view(batch_size, n_views, *alg_confidences.shape[1:])
            alg_confidences = alg_confidences / torch.clamp_min(alg_confidences.sum(dim=1, keepdim=True), min=1e-5)
            alg_confidences = alg_confidences + 1e-5  # for numerical stability

        # calcualte shapes
        batch_size, n_views, n_joints, heatmap_shape = heatmaps.shape[0], heatmaps.shape[1], heatmaps.shape[2], tuple(heatmaps.shape[3:])

        # upscale keypoints_2d, because image shape != heatmap shape
        keypoints_2d_transformed = torch.zeros_like(keypoints_2d_new)
        keypoints_2d_transformed[:, :, :, 0] = keypoints_2d_new[:, :, :, 0] * (image_shape[1] / heatmap_shape[1])
        keypoints_2d_transformed[:, :, :, 1] = keypoints_2d_new[:, :, :, 1] * (image_shape[0] / heatmap_shape[0])

        try:
            keypoints_3d = multiview.triangulate_batch_of_points(
                proj_matricies, keypoints_2d_transformed,
                confidences_batch=alg_confidences
            )
            keypoints_3d = keypoints_3d * 1000.0

        except RuntimeError as e:
            print("Error: ", e)
            # print("proj_matricies = ", proj_matricies)
            # print("keypoints_2d_batch_pred =", keypoints_2d_transformed)
            exit()
        # print(cropped_heatmap_gt.shape)
        # cropped_heatmap_gt = cropped_heatmap_gt.view(-1, *cropped_heatmap_gt.shape[-2:])
        # cropped_heatmap = cropped_heatmap.view(-1, *cropped_heatmap.shape[-2:])
        return keypoints_3d, keypoints_3d_pre, keypoints_2d_new * 4.0, heatmaps, alg_confidences

    def generate_target(self, keypoints_2d_gt, keypoints_2d_mean, image_size, heatmap_size, cropped_heatmap_size):
        '''
        :param joints:  [batch, num_joints, 2]
        :return: target
        '''

        feat_stride = (image_size / heatmap_size).cuda()
        mu_x = (keypoints_2d_gt[:, :, 0] / feat_stride[0] + 0.5).type(torch.long) - keypoints_2d_mean[:, :, 0] + cropped_heatmap_size[1] // 2
        mu_y = (keypoints_2d_gt[:, :, 1] / feat_stride[1] + 0.5).type(torch.long) - keypoints_2d_mean[:, :, 1] + cropped_heatmap_size[0] // 2
        keypoints_validity = (mu_x < cropped_heatmap_size[1]).float() * (mu_x >= 0).float() * (mu_y < cropped_heatmap_size[0]).float() * (mu_y >= 0).float()

        tmp_size = self.sigma * 3
        target = torch.zeros((keypoints_2d_gt.shape[0], keypoints_2d_gt.shape[1],
                           cropped_heatmap_size[0] + 2 * tmp_size,
                           cropped_heatmap_size[1] + 2 * tmp_size),
                          dtype=torch.float32).cuda()
        target = target.view(*target.shape[:2], -1)
        # Generate gaussian
        size = 2 * tmp_size + 1
        x = torch.arange(start=0, end=size, step=1, out=None).reshape(1, size)
        x = x.repeat(size, 1).cuda()
        y = x.transpose(0, 1)
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = torch.exp((-((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2)).float())
        g = g.unsqueeze(0).unsqueeze(0).repeat(keypoints_2d_gt.shape[0], keypoints_2d_gt.shape[1], 1, 1).view(keypoints_2d_gt.shape[0], keypoints_2d_gt.shape[1], -1)
        index_x_before = (
                    torch.arange(start=0, end=size, step=1).cuda().reshape(1, 1, size).repeat(keypoints_2d_gt.shape[0],
                                                                                              keypoints_2d_gt.shape[1],
                                                                                              1) + mu_x.unsqueeze(
                -1).repeat(1, 1, size)).unsqueeze(-2).repeat(1, 1, size, 1)
        index_x = torch.min(torch.max(torch.tensor(0), index_x_before.cpu()), torch.tensor(cropped_heatmap_size[1] + 2 * tmp_size - 1)).cuda()
        index_y_before = (
                    torch.arange(start=0, end=size, step=1).cuda().reshape(1, 1, size).repeat(keypoints_2d_gt.shape[0],
                                                                                              keypoints_2d_gt.shape[1],
                                                                                                    1) + mu_y.unsqueeze(
                -1).repeat(1, 1, size)).unsqueeze(-1).repeat(1, 1, 1, size)
        index_y = torch.min(torch.max(torch.tensor(0), index_y_before.cpu()), torch.tensor(cropped_heatmap_size[0] + 2 * tmp_size - 1)).cuda()
        index_y *= cropped_heatmap_size[1] + 2 * tmp_size
        index = (index_x + index_y).view(keypoints_2d_gt.shape[0], keypoints_2d_gt.shape[1], -1)
        # print(target.shape, index.shape, g.shape)
        target.scatter_(-1, index, g)
        target = target.view(keypoints_2d_gt.shape[0], keypoints_2d_gt.shape[1], cropped_heatmap_size[0] + 2*tmp_size, cropped_heatmap_size[1] + 2*tmp_size)
        target = target[:, :, tmp_size: -tmp_size, tmp_size: -tmp_size]

        return target, keypoints_validity


class AlgebraicTriangulationNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.use_confidences = config.model.use_confidences
        self.agg_type = config.model.agg_type
        self.backbone_type = config.model.backbone.name
        self.is_transformer = config.model.is_transformer
        self.motion_guide = config.model.motion_guide
        self.mask_subsequent = config.model.mask_subsequent
        self.channel = 16
        if self.agg_type != 'base':
            if self.backbone_type == 'resnet152':
                self.conv = nn.Sequential(nn.Conv2d(256, self.channel, kernel_size=(1, 1)), nn.BatchNorm2d(self.channel), nn.ReLU())
            else:
                self.conv = nn.Sequential(nn.Conv2d(32, self.channel, kernel_size=(1, 1)), nn.BatchNorm2d(self.channel), nn.ReLU())
        config.model.backbone.alg_confidences = False
        config.model.backbone.vol_confidences = False

        if self.use_confidences:
            config.model.backbone.alg_confidences = True
        if config.model.backbone.name == 'hrnet' and config.image_shape[0] == 256:
            self.backbone = pose_hrnet.get_pose_net(config, is_train=True, mask=config.mask)
            state_dict = torch.load(config.TEST.MODEL_FILE)
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[k.replace('backbone.', '')] = v
            self.backbone.load_state_dict(new_state_dict, strict=False)
            if self.agg_type == 'base':
                for p in self.backbone.parameters():
                    p.requires_grad = True
            else:
                for p in self.backbone.parameters():
                    p.requires_grad = False
            if self.use_confidences and self.agg_type == 'base':
                for p in self.backbone.alg_confidences.parameters():
                    p.requires_grad = True

        else:
            self.backbone = pose_resnet.get_pose_net(config.model.backbone)
            if self.agg_type == 'base':
                for p in self.backbone.parameters():
                    p.requires_grad = True
            else:
                for p in self.backbone.parameters():
                    p.requires_grad = False
            if self.use_confidences and self.agg_type == 'base':
                for p in self.backbone.alg_confidences.parameters():
                    p.requires_grad = True
        self.heatmap_softmax = config.model.heatmap_softmax
        self.heatmap_multiplier = config.model.heatmap_multiplier
        self.sigma = config.model.sigma
        self.pad_len = 8
        if self.agg_type == 'temporal':
            self.temporal_net = TemporalNet(17, 17)
        elif self.agg_type == 'graph':
            self.graph_net = st_gcn_heatmap.Model(use_confidences=self.use_confidences, is_transformer=self.is_transformer)
        elif self.agg_type == 'transformer':
            self.transformer_net = st_gcn_heatmap.Transformer(use_confidences=self.use_confidences, mask_subsequent=self.mask_subsequent)
        self.fill_back = True
        self.warp = config.warp
        if self.warp:
            self.stn = STN()
            for p in self.stn.parameters():
                p.requires_grad = True

        if self.motion_guide:
            self.motion = MotionNet(num_joints=17)
            state_dict = torch.load(os.path.join('/home/xueyz/Learnable-Triangulation/save_smpl', 'motion_model_smpl.pth'), map_location='cpu')
            new_state_dict = {}
            for keys, values in state_dict.items():
                keys = keys.replace('module.', '')
                new_state_dict[keys] = values
            self.motion.load_state_dict(new_state_dict, strict=False)
            print(f'Loading checkpoint from ' + '/home/xueyz/Learnable-Triangulation/save_smpl')
            # self.embedding = nn.Linear(17*3, 256)
            # self.position = PositionalEncoding(d_model=256, dropout=0.1)
            # self.motion = nn.Transformer(d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
            #                              dim_feedforward=1024, dropout=0.1, activation='gelu')
            # self.decoder = nn.Linear(256, 17*3)
            # self.motion_disturb = config.motion_disturb if hasattr(config, "motion_disturb") else 0.0


    def forward(self, images, K, proj_matricies, keypoints_2d_gt, frames=8, svd=True, vis=None, mask=None, keypoints_3d_history=None):
        # print(images.shape)  # [batch*frames, 4, 3, 384, 384]
        # keypoints_3d_history [batch, frames-1, 17, 3]
        batch_size, n_views = images.shape[:2]
        image_size = torch.from_numpy(np.array(images.shape[-2:]))
        # forward backbone and integral
        # reshape n_views dimension to batch dimension
        images = images.view(-1, *images.shape[2:])  # 64x3x384x384
        heatmaps, features, alg_confidences_history = self.backbone(images)
        # mask = None
        if mask is not None:
            mask = (mask.sum(2, keepdim=True) > 0.0).type(torch.float32)
            mask = Fu.interpolate(mask.view(-1, *mask.shape[2:]), (heatmaps.shape[-2], heatmaps.shape[-1]), mode='bilinear')
            heatmaps = heatmaps * mask
        if self.backbone_type == 'resnet152':
            heatmaps = heatmaps[:, (6, 2, 1, 0, 3, 4, 5, 7, 8, 16, 9, 13, 14, 15, 12, 11, 10), :, :]
            alg_confidences_history = alg_confidences_history[:, (6, 2, 1, 0, 3, 4, 5, 7, 8, 16, 9, 13, 14, 15, 12, 11, 10)]

        heatmaps_before_softmax = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])

        heatmap_size = torch.from_numpy(np.array(heatmaps_before_softmax.shape[3:]))
        keypoints_2d_origin, _ = op.integrate_tensor_2d(heatmaps * self.heatmap_multiplier, self.heatmap_softmax)  # 64x17x2
        if self.agg_type != 'base':
            features = self.conv(features)
            # print(features.shape)  # 64x16x64x64
            features = features.view(batch_size, n_views, *features.shape[1:]).transpose(1, 2).contiguous().transpose(0,
                                                                                                                      1).contiguous()
            features = features.view(-1, *features.shape[2:])
            features = features.unsqueeze(2).repeat(1, 1, 17, 1, 1)  # 16B x n_views x 17 x H x W
            keypoints_2d = keypoints_2d_origin.view(batch_size // frames, frames, n_views, -1, 2)  # [B, T, n_views, 17, 2] = [2, 8, 4, 17, 2]
            position_matrix = keypoints_2d.transpose(1, 2).contiguous().cuda()  # [B, n_views, T, 17, 2] = [2, 4, 8, 17, 2]
            position_matrix = position_matrix.view(-1, *position_matrix.shape[2:]).transpose(1, 2).contiguous().permute(3, 0, 1, 2).contiguous()  # [2, Bxn_views, 17, T] = [2, 8, 17, 8]
            position_matrix = position_matrix.reshape(*position_matrix.shape, 1, 1).repeat(1, 1, 1, 1, 2 * self.pad_len, 1).repeat(1, 1, 1, 1, 1, 2 * self.pad_len)  # [2, Bxn_views, 17, T, 16, 16] = [2, 8, 17, 8, 16, 16]
            H_range_matrix = (torch.arange(-self.pad_len, self.pad_len).cuda() + 0.5).reshape(1, 1, 1, 2 * self.pad_len, 1).repeat(*position_matrix.shape[1:4], 1, position_matrix.shape[-1])
            W_range_matrix = (torch.arange(-self.pad_len, self.pad_len).cuda() + 0.5).reshape(1, 1, 1, 1, 2 * self.pad_len).repeat(*position_matrix.shape[1:4], position_matrix.shape[-1], 1)
            position_matrix += torch.stack([W_range_matrix, H_range_matrix], dim=0)
            keypoints_2d = keypoints_2d.mean(dim=1, keepdim=True).long().repeat(1, frames, 1, 1, 1)
            keypoints_2d = keypoints_2d.view(-1, *keypoints_2d.shape[2:])
            K_ones = torch.ones(batch_size, n_views, 17).cuda()
            K_zeros = torch.zeros(batch_size, n_views, 17).cuda()
            K1 = torch.stack([0.25*K_ones, K_zeros, -keypoints_2d[..., 0].float(), K_zeros, 0.25*K_ones, -keypoints_2d[..., 1].float(), K_zeros, K_zeros, K_ones], dim=-1).view(batch_size, n_views, 17, 3, 3)


            pad_len = self.pad_len
            pad_feature = Fu.pad(features, [pad_len, pad_len,pad_len, pad_len])
            keypoints_2d_x = keypoints_2d[:, :, :, 0:1].repeat(self.channel, 1, 1, 2*pad_len).long() + torch.arange(0, 2*pad_len).view(1, 1, 1, -1).cuda()
            keypoints_2d_x = keypoints_2d_x.unsqueeze(3).repeat(1, 1, 1, pad_feature.shape[-2], 1)
            cropped_feature_x = torch.gather(pad_feature, 4, keypoints_2d_x)
            keypoints_2d_y = keypoints_2d[:, :, :, 1:2].repeat(self.channel, 1, 1, 2 * pad_len).long() + torch.arange(0, 2*pad_len).view(
                1, 1, 1, -1).cuda()
            keypoints_2d_y = keypoints_2d_y.unsqueeze(4).repeat(1, 1, 1, 1, 2 * pad_len)

            cropped_feature = torch.gather(cropped_feature_x, 3, keypoints_2d_y)

            cropped_feature = cropped_feature.view(self.channel, batch_size // frames, frames, n_views,
                                                   *cropped_feature.shape[2:]).transpose(3, 2).contiguous().transpose(3, 4).contiguous()
            cropped_feature = cropped_feature.view(self.channel, -1, *cropped_feature.shape[3:])  # (C, B * n_views, 17, T, H, W) = [16, 8, 17, 8, 16, 16]
            if self.agg_type == 'temporal':
                if self.use_confidences:
                    cropped_heatmap, alg_confidences = self.temporal_net(cropped_feature)
                    cropped_heatmap = cropped_heatmap.transpose(2, 1).contiguous().view(-1, cropped_heatmap.shape[1], *cropped_heatmap.shape[-2:])
                else:
                    cropped_heatmap, alg_confidences = self.temporal_net(cropped_feature)
                    cropped_heatmap = cropped_heatmap.transpose(2, 1).contiguous().view(-1, cropped_heatmap.shape[1], *cropped_heatmap.shape[-2:])
                    alg_confidences = torch.ones_like(alg_confidences).cuda()
            elif self.agg_type == 'graph' or self.agg_type == 'transformer':
                cropped_feature = cropped_feature.view(self.channel, batch_size//frames, n_views, *cropped_feature.shape[2:]).transpose(3, 4).contiguous()
                cropped_feature = cropped_feature.permute(1, 2, 3, 4, 0, 5, 6)  # [2, 4, 8, 17, 16, 16, 16] --> [2, 4, 8, 17, 18, 16, 16]
                position_matrix = position_matrix.view(position_matrix.shape[0], position_matrix.shape[1]//n_views, n_views, *position_matrix.shape[2:], ).permute(1, 2, 4, 3, 0, 5, 6)
                # print(cropped_feature.dtype, position_matrix.dtype)
                cropped_feature = torch.cat((cropped_feature, position_matrix), dim=4)
                if self.warp:
                    K2 = self.stn(cropped_feature.view(-1, *cropped_feature.shape[-3:])).view(-1, n_views, frames, 17, 3, 3).transpose(1, 2).contiguous().view(-1, n_views, 17, 3, 3)  # B*T, Views, V, 3, 3
                    K_warp = torch.matmul(K2, K1)
                else:
                    K_warp = None
                if self.agg_type == 'graph':
                    cropped_heatmap, alg_confidences = self.graph_net(cropped_feature)  # (B, n-views, T, 17, 17, H, W)
                else:
                    cropped_heatmap, alg_confidences = self.transformer_net(cropped_feature)  # (B, n-views, T, 17, 17, H, W)
                cropped_heatmap = cropped_heatmap.transpose(3, 4).contiguous()
                # cropped_heatmap = cropped_heatmap.permute(4, 0, 1, 2, 3, 5, 6).contiguous()
                # cropped_heatmap = cropped_heatmap.view(-1, *cropped_heatmap.shape[2:])
                cropped_heatmap = cropped_heatmap.transpose(1, 2).contiguous().view(batch_size, n_views, *cropped_heatmap.shape[3:])
                alg_confidences = alg_confidences.view(batch_size // frames, n_views, frames, -1).transpose(1, 2).contiguous().view(batch_size*n_views, -1)
            else:
                raise Exception('agg_type must be one of temporal, graph or base')

            if self.fill_back:
                heatmaps_before_softmax = Fu.pad(heatmaps_before_softmax, [pad_len, pad_len,pad_len, pad_len])
                modified_heatmap = torch.zeros_like(heatmaps_before_softmax).cuda()
                modified_heatmap = modified_heatmap.view(*modified_heatmap.shape[:-2], -1)
                keypoints_2d_x = keypoints_2d_x[:batch_size, :, :, :2*pad_len, :].unsqueeze(2).repeat(1, 1, 17, 1, 1, 1)
                keypoints_2d_y = keypoints_2d_y[:batch_size, :, :, :, :].unsqueeze(2).repeat(1, 1, 17, 1, 1, 1)
                modified_index = keypoints_2d_x.view(*keypoints_2d_x.shape[:-3], -1) \
                                 + heatmaps_before_softmax.shape[-1] * keypoints_2d_y.view(*keypoints_2d_y.shape[:-3], -1)
                # gaussian = torch.ones_like(cropped_heatmap).cuda()
                # cropped_heatmap = torch.eye(17).half().cuda().view(1, 1, 17, 17, 1, 1).repeat(batch_size, n_views, 1, 1, 2*pad_len, 2*pad_len)
                # modified_index = modified_index.repeat(17, 1, 1, 1)
                modified_heatmap.scatter_add_(-1, modified_index, cropped_heatmap.view(*cropped_heatmap.shape[:-3], -1))
                mask = torch.zeros_like(modified_heatmap).cuda()
                mask.scatter_(-1, modified_index, value=1.0)
                mask = mask.view(*mask.shape[:-1], *heatmaps_before_softmax.shape[-2:])
                cropped_heatmap = mask * modified_heatmap.view(*modified_heatmap.shape[:-1], *heatmaps_before_softmax.shape[-2:]) + (1.0 - mask) * heatmaps_before_softmax
                cropped_heatmap = cropped_heatmap[:, :, :, pad_len:-pad_len, pad_len:-pad_len]
                cropped_heatmap = cropped_heatmap.view(-1, *cropped_heatmap.shape[2:])

            keypoints_2d_gt = keypoints_2d_gt.view(-1, keypoints_2d_gt.shape[3], keypoints_2d_gt.shape[4])
            if self.fill_back:
                keypoints_2d_mean = torch.ones_like(keypoints_2d_gt, dtype=torch.long).cuda() * cropped_heatmap.shape[-1] // 2
            else:
                keypoints_2d_mean = keypoints_2d.view(-1, *keypoints_2d.shape[-2:])
            cropped_heatmap_gt, keypoints_2d_validity = self.generate_target(keypoints_2d_gt, keypoints_2d_mean, image_size, heatmap_size, cropped_heatmap.shape[-2:])
            keypoints_3d_validity = (keypoints_2d_validity.view(-1, n_views, keypoints_2d_validity.shape[-1]).sum(1) == n_views).float()
            # return htmap, cropped_heatmap_gt.view(-1, *cropped_heatmap_gt.shape[2:])

            if self.motion_guide:
                if keypoints_3d_history is not None:
                    keypoints_3d_history = keypoints_3d_history.view(batch_size // frames, frames-1, 17*3).transpose(1, 0).contiguous()
                    keypoints_3d_history += self.motion_disturb * torch.randn_like(keypoints_3d_history).cuda()
                else:
                    keypoints_2d_history = keypoints_2d_origin.view(batch_size // frames, frames, n_views, *keypoints_2d_origin.shape[-2:])
                    # keypoints_2d_history = keypoints_2d_history[:, :-1, :, :, :].contiguous()
                    alg_confidences_history = alg_confidences_history.view(batch_size // frames, frames, n_views, 17)
                    keypoints_2d_history = keypoints_2d_history.view(-1, n_views, 17, 2) * 4.0
                    alg_confidences_history = alg_confidences_history.view(-1, n_views, *alg_confidences.shape[1:])
                    alg_confidences_history = alg_confidences_history / alg_confidences_history.sum(dim=1, keepdim=True)
                    alg_confidences_history = alg_confidences_history + 1e-5  # for numerical stability
                    proj_matricies_history = proj_matricies.view(-1, frames, n_views, 3, 4)
                    proj_matricies_history = proj_matricies_history.view(-1, n_views, 3, 4)
                    keypoints_3d_history = multiview.triangulate_batch_of_points(
                        proj_matricies_history, keypoints_2d_history,
                        confidences_batch=alg_confidences_history
                    )
                    keypoints_3d_history = keypoints_3d_history * 1000.0
                    keypoints_3d_history = keypoints_3d_history.view(batch_size // frames, frames,
                                                                     17, 3).detach()

                smpl_refine = self.motion(keypoints_3d_history[:, :-1, :, :])
                keypoints_refine = batch_transform(smpl_refine[:, :, :-1, :].view(-1, 17, 3),
                                                   keypoints_3d_history[:, :-1, :, :].contiguous().view(
                                                       -1, 17, 3), parents=parents)
                pred = keypoints_refine.view(keypoints_3d_history.shape[0], -1, 17, 3) + smpl_refine[:, :, -1,
                                                                                               :].unsqueeze(2)
                # pred_3d_loss = torch.sqrt(((pred - keypoints_3d_history[:, 1:, :, :])**2).sum(-1)).mean()
                # print('3d: ', pred_3d_loss.item())
                # src = self.embedding(keypoints_3d_history.view(-1, 51)).view(frames-1, batch_size // frames, -1)
                # src = self.position(src)
                # if self.mask_subsequent:
                #     src_mask = self.motion.generate_square_subsequent_mask(frames - 1).cuda()
                #     tgt_mask = self.motion.generate_square_subsequent_mask(frames - 1).cuda()
                #     mem_mask = self.motion.generate_square_subsequent_mask(frames - 1).cuda()
                # else:
                #     src_mask = torch.zeros(frames - 1, frames - 1).cuda().float()
                #     tgt_mask = torch.zeros(frames - 1, frames - 1).cuda().float()
                #     mem_mask = torch.zeros(frames - 1, frames - 1).cuda().float()
                # pred = self.motion(src=src, tgt=src, src_mask=src_mask,
                #                    tgt_mask=tgt_mask, memory_mask=mem_mask)  # frames-1, batch, 256
                # pred = self.decoder(pred.view(-1, pred.shape[-1]))\
                #     .view(frames-1, batch_size // frames, -1)\
                #     .transpose(1, 0).contiguous()\
                #     .view(batch_size // frames, frames - 1, 17, 3)  # batch, frames-1, 17, 3
                proj_pred = proj_matricies.view(batch_size // frames, frames, n_views, 3, 4)[:, 1:, :, :].contiguous()  # batch, frames-1, 4, 3, 4
                pred = pred.unsqueeze(2).repeat(1, 1, n_views, 1, 1).view(-1, 3)
                proj_pred = proj_pred.unsqueeze(3).repeat(1, 1, 1, 17, 1, 1).view(-1, 3, 4)
                pred_2d = multiview.project_3d_points_to_image_batch(proj_pred, pred / 1000.0, convert_back_to_euclidean=True)
                pred_2d = pred_2d.view(-1, 17, 2)
                keypoints_2d_mean_history = keypoints_2d_mean.view(batch_size // frames, frames, n_views, 17, 2)[:, 1:, :, :, :].contiguous()
                keypoints_2d_mean_history = keypoints_2d_mean_history.view(-1, 17, 2)
                # pred_loss = torch.sqrt(((pred_2d.view(batch_size // frames, frames - 1, n_views, 17, 2) - keypoints_2d_gt.view(batch_size // frames, frames, n_views, 17, 2)[:, 1:, :, :, :]) ** 2).sum(-1)).mean()
                # print('2d:', pred_loss.item())
                cropped_heatmap_prior, _ = self.generate_target(pred_2d, keypoints_2d_mean_history,
                                                                                 image_size, heatmap_size,
                                                                                 cropped_heatmap.shape[-2:])
                cropped_heatmap_prior = cropped_heatmap_prior.view(batch_size // frames, frames - 1, n_views, 17, *cropped_heatmap_prior.shape[-2:])
                cropped_heatmap_pad = torch.ones_like(cropped_heatmap_prior[:, 0:1, :, :, :, :]).cuda()
                cropped_heatmap_prior = torch.cat([cropped_heatmap_pad, cropped_heatmap_prior], dim=1).view(-1, 17, *cropped_heatmap_prior.shape[-2:])
                cropped_heatmap = cropped_heatmap * 0.0 + cropped_heatmap * cropped_heatmap_prior * 1.0 + cropped_heatmap_prior * 0.0

            keypoints_2d_new, _ = op.integrate_tensor_2d(cropped_heatmap * self.heatmap_multiplier, self.heatmap_softmax)
            keypoints_2d_new = keypoints_2d_new.view(-1, n_views, *keypoints_2d_new.shape[-2:])
            keypoints_2d_new += keypoints_2d_mean.view(-1, n_views, *keypoints_2d_mean.shape[-2:]) - (torch.ones_like(keypoints_2d) * cropped_heatmap.shape[-1] // 2).cuda().float()
        else:
            keypoints_2d_new = keypoints_2d_origin.view(-1, n_views, *keypoints_2d_origin.shape[-2:])
            keypoints_2d_gt = keypoints_2d_gt.view(-1, *keypoints_2d_gt.shape[-2:])
            cropped_heatmap = heatmaps.clone()
            keypoints_2d_mean = torch.ones_like(keypoints_2d_gt, dtype=torch.long).cuda() * cropped_heatmap.shape[-1] // 2
            cropped_heatmap_gt, keypoints_2d_validity = self.generate_target(keypoints_2d_gt, keypoints_2d_mean, image_size, heatmap_size,
                                                    cropped_heatmap.shape[-2:])
            keypoints_3d_validity = (keypoints_2d_validity.view(-1, n_views, keypoints_2d_validity.shape[-1]).sum(
                1) == n_views).float()
            alg_confidences = alg_confidences_history

        # reshape back
        images = images.view(batch_size, n_views, *images.shape[1:])
        heatmaps = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])
        if alg_confidences is not None:
            alg_confidences = alg_confidences.view(batch_size, n_views, *alg_confidences.shape[1:])
            if vis is not None:
                alg_confidences = alg_confidences * vis
            # var_weight = multiview.get_heat_variance(cropped_heatmap.view(batch_size, n_views, *cropped_heatmap.shape[1:]))
            # alg_confidences = alg_confidences * var_weight * 4.0
            # norm confidences
            alg_confidences = alg_confidences / torch.clamp_min(alg_confidences.sum(dim=1, keepdim=True), min=1e-5)
            # alg_confidences = alg_confidences / alg_confidences.sum(dim=1, keepdim=True)
            alg_confidences = alg_confidences + 1e-5  # for numerical stability

        # calcualte shapes
        image_shape = tuple(images.shape[3:])
        batch_size, n_views, n_joints, heatmap_shape = heatmaps.shape[0], heatmaps.shape[1], heatmaps.shape[2], tuple(heatmaps.shape[3:])

        # change camera intrinsics
        new_K = torch.zeros_like(K)
        for batch_i in range(batch_size):
            for view_i in range(n_views):
                new_K[batch_i][view_i] = update_after_resize(K[batch_i][view_i], image_shape, heatmap_shape)

        proj_matricies_new = proj_matricies.float().cuda()

        # upscale keypoints_2d, because image shape != heatmap shape
        keypoints_2d_transformed = torch.zeros_like(keypoints_2d_new)
        keypoints_2d_transformed[:, :, :, 0] = keypoints_2d_new[:, :, :, 0] * (image_shape[1] / heatmap_shape[1])
        keypoints_2d_transformed[:, :, :, 1] = keypoints_2d_new[:, :, :, 1] * (image_shape[0] / heatmap_shape[0])
        # triangulate
        # print(proj_matricies.dtype, keypoints_2d_transformed.dtype)
        # print('proj_mat:', proj_matricies[0, 0, :, :])
        # print('kp2d:', keypoints_2d_transformed[0, 0, 0, :])
        try:
            if not svd:
                keypoints_2d_origin = keypoints_2d_origin.view(batch_size, n_views, *keypoints_2d_origin.shape[1:])
                keypoints_2d_transformed = 0.0 * keypoints_2d_transformed + keypoints_2d_origin * 4
                # print(alg_confidences)
            if self.agg_type != 'base' and self.warp:
                proj_matricies = torch.matmul(K_warp, proj_matricies.unsqueeze(2).repeat(1, 1, 17, 1, 1))
                keypoints_2d_transformed = torch.matmul(K_warp, torch.cat([keypoints_2d_transformed, torch.ones(batch_size, n_views, 17, 1).cuda()], dim=-1).unsqueeze(-1))
                keypoints_2d_transformed = keypoints_2d_transformed[:, :, :, :2, 0]
                keypoints_3d = multiview.triangulate_batch_of_points_warping(
                    proj_matricies, keypoints_2d_transformed,
                    confidences_batch=alg_confidences
                )
            else:
                keypoints_3d = multiview.triangulate_batch_of_points(
                    proj_matricies, keypoints_2d_transformed,
                    confidences_batch=alg_confidences
                )
            keypoints_3d = keypoints_3d * 1000.0
        except RuntimeError as e:
            print("Error: ", e)
            # print("proj_matricies = ", proj_matricies)
            # print("keypoints_2d_batch_pred =", keypoints_2d_transformed)
            exit()
        # print(cropped_heatmap_gt.shape)
        # cropped_heatmap_gt = cropped_heatmap_gt.view(-1, *cropped_heatmap_gt.shape[-2:])
        # cropped_heatmap = cropped_heatmap.view(-1, *cropped_heatmap.shape[-2:])
        if self.agg_type == 'base':
            return keypoints_3d, keypoints_2d_new * 4.0, cropped_heatmap, proj_matricies_new, new_K, keypoints_2d_transformed, cropped_heatmap_gt, heatmaps, keypoints_2d_validity, keypoints_3d_validity
        else:
            return keypoints_3d, keypoints_2d_new * 4.0, cropped_heatmap, proj_matricies_new, new_K, keypoints_2d_transformed, cropped_heatmap_gt, heatmaps, keypoints_2d_validity, keypoints_3d_validity

    def generate_target(self, keypoints_2d_gt, keypoints_2d_mean, image_size, heatmap_size, cropped_heatmap_size):
        '''
        :param joints:  [batch, num_joints, 2]
        :return: target
        '''

        feat_stride = (image_size / heatmap_size).cuda()
        mu_x = (keypoints_2d_gt[:, :, 0] / feat_stride[0] + 0.5).type(torch.long) - keypoints_2d_mean[:, :, 0] + cropped_heatmap_size[1] // 2
        mu_y = (keypoints_2d_gt[:, :, 1] / feat_stride[1] + 0.5).type(torch.long) - keypoints_2d_mean[:, :, 1] + cropped_heatmap_size[0] // 2
        keypoints_validity = (mu_x < cropped_heatmap_size[1]).float() * (mu_x >= 0).float() * (mu_y < cropped_heatmap_size[0]).float() * (mu_y >= 0).float()

        tmp_size = self.sigma * 3
        target = torch.zeros((keypoints_2d_gt.shape[0], keypoints_2d_gt.shape[1],
                           cropped_heatmap_size[0] + 2 * tmp_size,
                           cropped_heatmap_size[1] + 2 * tmp_size),
                          dtype=torch.float32).cuda()
        target = target.view(*target.shape[:2], -1)
        # Generate gaussian
        size = 2 * tmp_size + 1
        x = torch.arange(start=0, end=size, step=1, out=None).reshape(1, size)
        x = x.repeat(size, 1).cuda()
        y = x.transpose(0, 1)
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = torch.exp((-((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2)).float())
        g = g.unsqueeze(0).unsqueeze(0).repeat(keypoints_2d_gt.shape[0], keypoints_2d_gt.shape[1], 1, 1).view(keypoints_2d_gt.shape[0], keypoints_2d_gt.shape[1], -1)
        index_x_before = (
                    torch.arange(start=0, end=size, step=1).cuda().reshape(1, 1, size).repeat(keypoints_2d_gt.shape[0],
                                                                                              keypoints_2d_gt.shape[1],
                                                                                              1) + mu_x.unsqueeze(
                -1).repeat(1, 1, size)).unsqueeze(-2).repeat(1, 1, size, 1)
        index_x = torch.min(torch.max(torch.tensor(0), index_x_before.cpu()), torch.tensor(cropped_heatmap_size[1] + 2 * tmp_size - 1)).cuda()
        index_y_before = (
                    torch.arange(start=0, end=size, step=1).cuda().reshape(1, 1, size).repeat(keypoints_2d_gt.shape[0],
                                                                                              keypoints_2d_gt.shape[1],
                                                                                                    1) + mu_y.unsqueeze(
                -1).repeat(1, 1, size)).unsqueeze(-1).repeat(1, 1, 1, size)
        index_y = torch.min(torch.max(torch.tensor(0), index_y_before.cpu()), torch.tensor(cropped_heatmap_size[0] + 2 * tmp_size - 1)).cuda()
        index_y *= cropped_heatmap_size[1] + 2 * tmp_size
        index = (index_x + index_y).view(keypoints_2d_gt.shape[0], keypoints_2d_gt.shape[1], -1)
        # print(target.shape, index.shape, g.shape)
        target.scatter_(-1, index, g)
        target = target.view(keypoints_2d_gt.shape[0], keypoints_2d_gt.shape[1], cropped_heatmap_size[0] + 2*tmp_size, cropped_heatmap_size[1] + 2*tmp_size)
        target = target[:, :, tmp_size: -tmp_size, tmp_size: -tmp_size]

        return target, keypoints_validity


class AlgDepth(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.use_confidences = config.model.use_confidences
        self.backbone_type = config.model.backbone.name
        self.channel = 16
        self.pad_len = 8
        self.depthnet = DepthNet(self.channel, window=self.pad_len*2, depth_num=20, depth_interval=4)
        config.model.backbone.alg_confidences = False
        config.model.backbone.vol_confidences = False
        if self.use_confidences:
            config.model.backbone.alg_confidences = True
        self.backbone = pose_resnet.get_pose_net(config.model.backbone)
        self.heatmap_softmax = config.model.heatmap_softmax
        self.heatmap_multiplier = config.model.heatmap_multiplier
        self.sigma = config.model.sigma

    def forward(self, images, proj_matricies, keypoints_2d_gt):
        # print(images.shape)  # [batch, 4, 3, 384, 384]
        batch_size, n_views = images.shape[:2]
        image_size = torch.from_numpy(np.array(images.shape[-2:]))
        # forward backbone and integral
        # reshape n_views dimension to batch dimension
        images = images.view(-1, *images.shape[2:])  # 64x3x384x384
        heatmaps, features, alg_confidences_history = self.backbone(images)
        heatmaps = heatmaps[:, (6, 2, 1, 0, 3, 4, 5, 7, 8, 16, 9, 13, 14, 15, 12, 11, 10), :, :]
        alg_confidences_history = alg_confidences_history[:, (6, 2, 1, 0, 3, 4, 5, 7, 8, 16, 9, 13, 14, 15, 12, 11, 10)]

        heatmaps_before_softmax = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])

        heatmap_size = torch.from_numpy(np.array(heatmaps_before_softmax.shape[3:]))
        keypoints_2d_origin, _ = op.integrate_tensor_2d(heatmaps * self.heatmap_multiplier, self.heatmap_softmax)  # 64x17x2

        keypoints_2d_new = keypoints_2d_origin.view(-1, n_views, *keypoints_2d_origin.shape[-2:])
        keypoints_2d_gt = keypoints_2d_gt.view(-1, *keypoints_2d_gt.shape[-2:])
        cropped_heatmap = heatmaps.clone()
        keypoints_2d_mean = torch.ones_like(keypoints_2d_gt, dtype=torch.long).cuda() * cropped_heatmap.shape[-1] // 2
        cropped_heatmap_gt, _ = self.generate_target(keypoints_2d_gt, keypoints_2d_mean, image_size, heatmap_size,
                                                cropped_heatmap.shape[-2:])
        alg_confidences = alg_confidences_history

        # reshape back
        images = images.view(batch_size, n_views, *images.shape[1:])
        heatmaps = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])  # (B, views, 17, H, W)
        if alg_confidences is not None:
            alg_confidences = alg_confidences.view(batch_size, n_views, *alg_confidences.shape[1:])
            # norm confidences
            alg_confidences = alg_confidences / torch.clamp_min(alg_confidences.sum(dim=1, keepdim=True), min=1e-5)
            alg_confidences = alg_confidences + 1e-5  # for numerical stability

        # calcualte shapes
        image_shape = tuple(images.shape[3:])
        batch_size, n_views, n_joints, heatmap_shape = heatmaps.shape[0], heatmaps.shape[1], heatmaps.shape[2], tuple(heatmaps.shape[3:])

        # upscale keypoints_2d, because image shape != heatmap shape
        keypoints_2d_transformed = torch.zeros_like(keypoints_2d_new)
        keypoints_2d_transformed[:, :, :, 0] = keypoints_2d_new[:, :, :, 0] * (image_shape[1] / heatmap_shape[1])
        keypoints_2d_transformed[:, :, :, 1] = keypoints_2d_new[:, :, :, 1] * (image_shape[0] / heatmap_shape[0])

        keypoints_3d = multiview.triangulate_batch_of_points(
            proj_matricies, keypoints_2d_transformed,
            confidences_batch=alg_confidences
        )
        keypoints_3d = keypoints_3d * 1000.0

        features = features.view(batch_size, n_views, *features.shape[1:])
        init_depth = get_init_depth(keypoints_3d, proj_matricies)
        depth = self.depthnet(features, proj_matricies, keypoints_2d_transformed, init_depth)  # (B, views, 17, h, w)
        idx = create_idx(proj_matricies, keypoints_2d_transformed, depth, window=self.pad_len*2)  # (B, views, 17, views, h, w, 2)
        idx = idx.permute(0, 3, 2, 1, 4, 5, 6).contiguous()
        idx = idx.view(batch_size*n_views*17, n_views*self.pad_len*2, -1, 2)
        idx = 2.0 * idx / heatmaps.shape[-1] - 1.0
        heatmaps_sample = heatmaps.view(batch_size*n_views*17, 1, *heatmaps.shape[-2:])
        heatmaps_sample = Fu.grid_sample(heatmaps_sample, idx, mode='bilinear')  # (B*views*17, 1, views*h, w)
        heatmaps_sample = heatmaps_sample.view(batch_size, n_views, 17, n_views, self.pad_len*2, self.pad_len*2)
        heatmaps_sample = heatmaps_sample.mean(1).view(batch_size, 17, n_views, self.pad_len*2*self.pad_len*2)  # (B*17*views, h*w)
        heatmaps_sample = heatmaps_sample.transpose(2, 1).contiguous().view(batch_size*n_views*17, -1)  # (B*views*17, h*w)


        keypoint_2d_x = keypoints_2d_new[:, :, :, 0:1].long()
        keypoint_2d_y = keypoints_2d_new[:, :, :, 1:2].long()
        shift_x = torch.arange(0, self.pad_len * 2).cuda().long().view(1, 1, self.pad_len * 2).repeat(1, self.pad_len * 2, 1) - self.pad_len
        shift_y = torch.arange(0, self.pad_len * 2).cuda().long().view(1, self.pad_len * 2, 1).repeat(1, 1, self.pad_len * 2) - self.pad_len
        keypoint_2d_x = keypoint_2d_x.view(-1, 1, 1).repeat(1, self.pad_len * 2, self.pad_len * 2)
        keypoint_2d_y = keypoint_2d_y.view(-1, 1, 1).repeat(1, self.pad_len * 2, self.pad_len * 2)
        keypoint_2d_x = keypoint_2d_x + shift_x
        keypoint_2d_y = keypoint_2d_y + shift_y
        keypoint_idx = keypoint_2d_x + keypoint_2d_y * heatmap_shape[1]
        keypoint_idx = keypoint_idx.view(-1, self.pad_len*2*self.pad_len*2)  # (B*views*17, h*w)
        heatmaps_pred = torch.zeros(batch_size, n_views, 17, heatmap_shape[0], heatmap_shape[1]).cuda().float()
        heatmaps_pred = heatmaps_pred.view(-1, heatmap_shape[0]*heatmap_shape[1])
        keypoint_idx = torch.clamp_max(torch.clamp_min(keypoint_idx, min=0), max=heatmap_shape[0]*heatmap_shape[1]-1)
        heatmaps_pred.scatter_(-1, keypoint_idx, heatmaps_sample)
        heatmaps_pred = heatmaps_pred.view(batch_size*n_views, 17, heatmap_shape[0], heatmap_shape[1])

        depth_pred = torch.zeros(batch_size*n_views*17, heatmap_shape[0]*heatmap_shape[1]).cuda().float()
        depth_sample = depth.view(batch_size*n_views*17, self.pad_len*2*self.pad_len*2)
        depth_pred.scatter_(-1, keypoint_idx, depth_sample)
        depth_pred = depth_pred.view(batch_size*n_views*17, 1, heatmap_shape[0],heatmap_shape[1])  # (B*views*17, 1, H, W)
        depth_pred = Fu.grid_sample(depth_pred, idx, mode='bilinear')  # (B*views*17, 1, views*h, w)

        depth_pred = depth_pred.view(batch_size, n_views, 17, n_views, self.pad_len * 2, self.pad_len * 2)
        depth_pred = depth_pred.permute(0, 2, 3, 1, 4, 5).contiguous()

        keypoints_2d_pred, _ = op.integrate_tensor_2d(heatmaps_pred * self.heatmap_multiplier, self.heatmap_softmax)  # 64x17x2
        keypoints_2d_pred = keypoints_2d_pred.view(batch_size, n_views, 17, 2) * 4.0
        keypoints_3d_pred = multiview.triangulate_batch_of_points(
            proj_matricies, keypoints_2d_pred,
            confidences_batch=alg_confidences
        )
        keypoints_3d_pred = keypoints_3d_pred * 1000.0

        return keypoints_3d_pred, keypoints_2d_pred, cropped_heatmap, cropped_heatmap_gt, depth_pred, heatmaps_pred


    def generate_target(self, keypoints_2d_gt, keypoints_2d_mean, image_size, heatmap_size, cropped_heatmap_size):
        '''
        :param joints:  [batch, num_joints, 2]
        :return: target
        '''

        feat_stride = (image_size / heatmap_size).cuda()
        mu_x = (keypoints_2d_gt[:, :, 0] / feat_stride[0] + 0.5).type(torch.long) - keypoints_2d_mean[:, :, 0] + cropped_heatmap_size[1] // 2
        mu_y = (keypoints_2d_gt[:, :, 1] / feat_stride[1] + 0.5).type(torch.long) - keypoints_2d_mean[:, :, 1] + cropped_heatmap_size[0] // 2
        keypoints_validity = (mu_x < cropped_heatmap_size[1]).float() * (mu_x >= 0).float() * (mu_y < cropped_heatmap_size[0]).float() * (mu_y >= 0).float()

        tmp_size = self.sigma * 3
        target = torch.zeros((keypoints_2d_gt.shape[0], keypoints_2d_gt.shape[1],
                           cropped_heatmap_size[0] + 2 * tmp_size,
                           cropped_heatmap_size[1] + 2 * tmp_size),
                          dtype=torch.float32).cuda()
        target = target.view(*target.shape[:2], -1)
        # Generate gaussian
        size = 2 * tmp_size + 1
        x = torch.arange(start=0, end=size, step=1, out=None).reshape(1, size)
        x = x.repeat(size, 1).cuda()
        y = x.transpose(0, 1)
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = torch.exp((-((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2)).float())
        g = g.unsqueeze(0).unsqueeze(0).repeat(keypoints_2d_gt.shape[0], keypoints_2d_gt.shape[1], 1, 1).view(keypoints_2d_gt.shape[0], keypoints_2d_gt.shape[1], -1)
        index_x_before = (
                    torch.arange(start=0, end=size, step=1).cuda().reshape(1, 1, size).repeat(keypoints_2d_gt.shape[0],
                                                                                              keypoints_2d_gt.shape[1],
                                                                                              1) + mu_x.unsqueeze(
                -1).repeat(1, 1, size)).unsqueeze(-2).repeat(1, 1, size, 1)
        index_x = torch.min(torch.max(torch.tensor(0), index_x_before.cpu()), torch.tensor(cropped_heatmap_size[1] + 2 * tmp_size - 1)).cuda()
        index_y_before = (
                    torch.arange(start=0, end=size, step=1).cuda().reshape(1, 1, size).repeat(keypoints_2d_gt.shape[0],
                                                                                              keypoints_2d_gt.shape[1],
                                                                                                    1) + mu_y.unsqueeze(
                -1).repeat(1, 1, size)).unsqueeze(-1).repeat(1, 1, 1, size)
        index_y = torch.min(torch.max(torch.tensor(0), index_y_before.cpu()), torch.tensor(cropped_heatmap_size[0] + 2 * tmp_size - 1)).cuda()
        index_y *= cropped_heatmap_size[1] + 2 * tmp_size
        index = (index_x + index_y).view(keypoints_2d_gt.shape[0], keypoints_2d_gt.shape[1], -1)
        # print(target.shape, index.shape, g.shape)
        target.scatter_(-1, index, g)
        target = target.view(keypoints_2d_gt.shape[0], keypoints_2d_gt.shape[1], cropped_heatmap_size[0] + 2*tmp_size, cropped_heatmap_size[1] + 2*tmp_size)
        target = target[:, :, tmp_size: -tmp_size, tmp_size: -tmp_size]

        return target, keypoints_validity




class AlgebraicTriangulationNet_test(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.use_confidences = config.model.use_confidences
        self.agg_type = config.model.agg_type
        self.backbone_type = config.model.backbone.name
        self.is_transformer = config.model.is_transformer
        self.motion_guide = config.model.motion_guide
        self.mask_subsequent = config.model.mask_subsequent
        self.channel = 16
        if self.agg_type != 'base':
            if self.backbone_type == 'resnet152':
                self.conv = nn.Sequential(nn.Conv2d(256, self.channel, kernel_size=(1, 1)), nn.BatchNorm2d(self.channel), nn.ReLU())
            else:
                self.conv = nn.Sequential(nn.Conv2d(32, self.channel, kernel_size=(1, 1)), nn.BatchNorm2d(self.channel), nn.ReLU())
        config.model.backbone.alg_confidences = False
        config.model.backbone.vol_confidences = False

        if self.use_confidences:
            config.model.backbone.alg_confidences = True
        if config.model.backbone.name == 'hrnet' and config.image_shape[0] == 256:
            self.backbone = pose_hrnet.get_pose_net(config, is_train=True, mask=config.mask)
            state_dict = torch.load(config.TEST.MODEL_FILE)
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[k.replace('backbone.', '')] = v
            self.backbone.load_state_dict(new_state_dict, strict=False)
            if self.agg_type == 'base':
                for p in self.backbone.parameters():
                    p.requires_grad = True
            else:
                for p in self.backbone.parameters():
                    p.requires_grad = False
            if self.use_confidences and self.agg_type == 'base':
                for p in self.backbone.alg_confidences.parameters():
                    p.requires_grad = True

        else:
            self.backbone = pose_resnet.get_pose_net(config.model.backbone)
            if self.agg_type == 'base':
                for p in self.backbone.parameters():
                    p.requires_grad = True
            else:
                for p in self.backbone.parameters():
                    p.requires_grad = False
            if self.use_confidences and self.agg_type == 'base':
                for p in self.backbone.alg_confidences.parameters():
                    p.requires_grad = True
        self.heatmap_softmax = config.model.heatmap_softmax
        self.heatmap_multiplier = config.model.heatmap_multiplier
        self.sigma = config.model.sigma
        self.pad_len = 8
        if self.agg_type == 'temporal':
            self.temporal_net = TemporalNet(17, 17)
        elif self.agg_type == 'graph':
            self.graph_net = st_gcn_heatmap.Model(use_confidences=self.use_confidences, is_transformer=self.is_transformer)
        elif self.agg_type == 'transformer':
            self.transformer_net = st_gcn_heatmap.Transformer(use_confidences=self.use_confidences, mask_subsequent=self.mask_subsequent)
        self.fill_back = True
        self.warp = config.warp
        if self.warp:
            self.stn = STN()
            for p in self.stn.parameters():
                p.requires_grad = True

        if self.motion_guide:
            self.motion = MotionNet(num_joints=17)
            state_dict = torch.load(os.path.join('/home/xueyz/Learnable-Triangulation/save_smpl', 'motion_model_smpl.pth'), map_location='cpu')
            new_state_dict = {}
            for keys, values in state_dict.items():
                keys = keys.replace('module.', '')
                new_state_dict[keys] = values
            self.motion.load_state_dict(new_state_dict, strict=False)
            print(f'Loading checkpoint from ' + '/home/xueyz/Learnable-Triangulation/save_smpl')
            # self.embedding = nn.Linear(17*3, 256)
            # self.position = PositionalEncoding(d_model=256, dropout=0.1)
            # self.motion = nn.Transformer(d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
            #                              dim_feedforward=1024, dropout=0.1, activation='gelu')
            # self.decoder = nn.Linear(256, 17*3)
            # self.motion_disturb = config.motion_disturb if hasattr(config, "motion_disturb") else 0.0


    def forward(self, images, K, proj_matricies, keypoints_2d_gt, frames=8, svd=True, vis=None, mask=None, keypoints_3d_history=None):
        # print(images.shape)  # [batch*frames, 4, 3, 384, 384]
        # keypoints_3d_history [batch, frames-1, 17, 3]
        batch_size, n_views = images.shape[:2]
        image_size = torch.from_numpy(np.array(images.shape[-2:]))
        # forward backbone and integral
        # reshape n_views dimension to batch dimension
        images = images.view(-1, *images.shape[2:])  # 64x3x384x384
        heatmaps, features, alg_confidences_history = self.backbone(images)
        # mask = None
        if mask is not None:
            mask = (mask.sum(2, keepdim=True) > 0.0).type(torch.float32)
            mask = Fu.interpolate(mask.view(-1, *mask.shape[2:]), (heatmaps.shape[-2], heatmaps.shape[-1]), mode='bilinear')
            heatmaps = heatmaps * mask
        if self.backbone_type == 'resnet152':
            heatmaps = heatmaps[:, (6, 2, 1, 0, 3, 4, 5, 7, 8, 16, 9, 13, 14, 15, 12, 11, 10), :, :]
            alg_confidences_history = alg_confidences_history[:, (6, 2, 1, 0, 3, 4, 5, 7, 8, 16, 9, 13, 14, 15, 12, 11, 10)]

        heatmaps_before_softmax = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])

        heatmap_size = torch.from_numpy(np.array(heatmaps_before_softmax.shape[3:]))
        keypoints_2d_origin, _ = op.integrate_tensor_2d(heatmaps * self.heatmap_multiplier, self.heatmap_softmax)  # 64x17x2
        if self.agg_type != 'base':
            features = self.conv(features)
            # print(features.shape)  # 64x16x64x64
            features = features.view(batch_size, n_views, *features.shape[1:]).transpose(1, 2).contiguous().transpose(0,
                                                                                                                      1).contiguous()
            features = features.view(-1, *features.shape[2:])
            features = features.unsqueeze(2).repeat(1, 1, 17, 1, 1)  # 16B x n_views x 17 x H x W
            keypoints_2d = keypoints_2d_origin.view(batch_size // frames, frames, n_views, -1, 2)  # [B, T, n_views, 17, 2] = [2, 8, 4, 17, 2]
            position_matrix = keypoints_2d.transpose(1, 2).contiguous().cuda()  # [B, n_views, T, 17, 2] = [2, 4, 8, 17, 2]
            position_matrix = position_matrix.view(-1, *position_matrix.shape[2:]).transpose(1, 2).contiguous().permute(3, 0, 1, 2).contiguous()  # [2, Bxn_views, 17, T] = [2, 8, 17, 8]
            position_matrix = position_matrix.reshape(*position_matrix.shape, 1, 1).repeat(1, 1, 1, 1, 2 * self.pad_len, 1).repeat(1, 1, 1, 1, 1, 2 * self.pad_len)  # [2, Bxn_views, 17, T, 16, 16] = [2, 8, 17, 8, 16, 16]
            H_range_matrix = (torch.arange(-self.pad_len, self.pad_len).cuda() + 0.5).reshape(1, 1, 1, 2 * self.pad_len, 1).repeat(*position_matrix.shape[1:4], 1, position_matrix.shape[-1])
            W_range_matrix = (torch.arange(-self.pad_len, self.pad_len).cuda() + 0.5).reshape(1, 1, 1, 1, 2 * self.pad_len).repeat(*position_matrix.shape[1:4], position_matrix.shape[-1], 1)
            position_matrix += torch.stack([W_range_matrix, H_range_matrix], dim=0)
            keypoints_2d = keypoints_2d.mean(dim=1, keepdim=True).long().repeat(1, frames, 1, 1, 1)
            keypoints_2d = keypoints_2d.view(-1, *keypoints_2d.shape[2:])
            K_ones = torch.ones(batch_size, n_views, 17).cuda()
            K_zeros = torch.zeros(batch_size, n_views, 17).cuda()
            K1 = torch.stack([0.25*K_ones, K_zeros, -keypoints_2d[..., 0].float(), K_zeros, 0.25*K_ones, -keypoints_2d[..., 1].float(), K_zeros, K_zeros, K_ones], dim=-1).view(batch_size, n_views, 17, 3, 3)


            pad_len = self.pad_len
            pad_feature = Fu.pad(features, [pad_len, pad_len,pad_len, pad_len])
            keypoints_2d_x = keypoints_2d[:, :, :, 0:1].repeat(self.channel, 1, 1, 2*pad_len).long() + torch.arange(0, 2*pad_len).view(1, 1, 1, -1).cuda()
            keypoints_2d_x = keypoints_2d_x.unsqueeze(3).repeat(1, 1, 1, pad_feature.shape[-2], 1)
            cropped_feature_x = torch.gather(pad_feature, 4, keypoints_2d_x)
            keypoints_2d_y = keypoints_2d[:, :, :, 1:2].repeat(self.channel, 1, 1, 2 * pad_len).long() + torch.arange(0, 2*pad_len).view(
                1, 1, 1, -1).cuda()
            keypoints_2d_y = keypoints_2d_y.unsqueeze(4).repeat(1, 1, 1, 1, 2 * pad_len)

            cropped_feature = torch.gather(cropped_feature_x, 3, keypoints_2d_y)

            cropped_feature = cropped_feature.view(self.channel, batch_size // frames, frames, n_views,
                                                   *cropped_feature.shape[2:]).transpose(3, 2).contiguous().transpose(3, 4).contiguous()
            cropped_feature = cropped_feature.view(self.channel, -1, *cropped_feature.shape[3:])  # (C, B * n_views, 17, T, H, W) = [16, 8, 17, 8, 16, 16]
            if self.agg_type == 'temporal':
                if self.use_confidences:
                    cropped_heatmap, alg_confidences = self.temporal_net(cropped_feature)
                    cropped_heatmap = cropped_heatmap.transpose(2, 1).contiguous().view(-1, cropped_heatmap.shape[1], *cropped_heatmap.shape[-2:])
                else:
                    cropped_heatmap, alg_confidences = self.temporal_net(cropped_feature)
                    cropped_heatmap = cropped_heatmap.transpose(2, 1).contiguous().view(-1, cropped_heatmap.shape[1], *cropped_heatmap.shape[-2:])
                    alg_confidences = torch.ones_like(alg_confidences).cuda()
            elif self.agg_type == 'graph' or self.agg_type == 'transformer':
                cropped_feature = cropped_feature.view(self.channel, batch_size//frames, n_views, *cropped_feature.shape[2:]).transpose(3, 4).contiguous()
                cropped_feature = cropped_feature.permute(1, 2, 3, 4, 0, 5, 6)  # [2, 4, 8, 17, 16, 16, 16] --> [2, 4, 8, 17, 18, 16, 16]
                position_matrix = position_matrix.view(position_matrix.shape[0], position_matrix.shape[1]//n_views, n_views, *position_matrix.shape[2:], ).permute(1, 2, 4, 3, 0, 5, 6)
                # print(cropped_feature.dtype, position_matrix.dtype)
                cropped_feature = torch.cat((cropped_feature, position_matrix), dim=4)
                if self.warp:
                    K2 = self.stn(cropped_feature.view(-1, *cropped_feature.shape[-3:])).view(-1, n_views, frames, 17, 3, 3).transpose(1, 2).contiguous().view(-1, n_views, 17, 3, 3)  # B*T, Views, V, 3, 3
                    K_warp = torch.matmul(K2, K1)
                else:
                    K_warp = None
                if self.agg_type == 'graph':
                    cropped_heatmap, alg_confidences = self.graph_net(cropped_feature)  # (B, n-views, T, 17, 17, H, W)
                else:
                    cropped_heatmap, alg_confidences = self.transformer_net(cropped_feature)  # (B, n-views, T, 17, 17, H, W)
                cropped_heatmap = cropped_heatmap.transpose(3, 4).contiguous()
                # cropped_heatmap = cropped_heatmap.permute(4, 0, 1, 2, 3, 5, 6).contiguous()
                # cropped_heatmap = cropped_heatmap.view(-1, *cropped_heatmap.shape[2:])
                cropped_heatmap = cropped_heatmap.transpose(1, 2).contiguous().view(batch_size, n_views, *cropped_heatmap.shape[3:])
                alg_confidences = alg_confidences.view(batch_size // frames, n_views, frames, -1).transpose(1, 2).contiguous().view(batch_size*n_views, -1)
            else:
                raise Exception('agg_type must be one of temporal, graph or base')

            if self.fill_back:
                heatmaps_before_softmax = Fu.pad(heatmaps_before_softmax, [pad_len, pad_len,pad_len, pad_len])
                modified_heatmap = torch.zeros_like(heatmaps_before_softmax).cuda()
                modified_heatmap = modified_heatmap.view(*modified_heatmap.shape[:-2], -1)
                keypoints_2d_x = keypoints_2d_x[:batch_size, :, :, :2*pad_len, :].unsqueeze(2).repeat(1, 1, 17, 1, 1, 1)
                keypoints_2d_y = keypoints_2d_y[:batch_size, :, :, :, :].unsqueeze(2).repeat(1, 1, 17, 1, 1, 1)
                modified_index = keypoints_2d_x.view(*keypoints_2d_x.shape[:-3], -1) \
                                 + heatmaps_before_softmax.shape[-1] * keypoints_2d_y.view(*keypoints_2d_y.shape[:-3], -1)
                # gaussian = torch.ones_like(cropped_heatmap).cuda()
                # cropped_heatmap = torch.eye(17).half().cuda().view(1, 1, 17, 17, 1, 1).repeat(batch_size, n_views, 1, 1, 2*pad_len, 2*pad_len)
                # modified_index = modified_index.repeat(17, 1, 1, 1)
                modified_heatmap.scatter_add_(-1, modified_index, cropped_heatmap.view(*cropped_heatmap.shape[:-3], -1))
                mask = torch.zeros_like(modified_heatmap).cuda()
                mask.scatter_(-1, modified_index, value=1.0)
                mask = mask.view(*mask.shape[:-1], *heatmaps_before_softmax.shape[-2:])
                cropped_heatmap = mask * modified_heatmap.view(*modified_heatmap.shape[:-1], *heatmaps_before_softmax.shape[-2:]) + (1.0 - mask) * heatmaps_before_softmax
                cropped_heatmap = cropped_heatmap[:, :, :, pad_len:-pad_len, pad_len:-pad_len]
                cropped_heatmap = cropped_heatmap.view(-1, *cropped_heatmap.shape[2:])

            keypoints_2d_gt = keypoints_2d_gt.view(-1, keypoints_2d_gt.shape[3], keypoints_2d_gt.shape[4])
            if self.fill_back:
                keypoints_2d_mean = torch.ones_like(keypoints_2d_gt, dtype=torch.long).cuda() * cropped_heatmap.shape[-1] // 2
            else:
                keypoints_2d_mean = keypoints_2d.view(-1, *keypoints_2d.shape[-2:])
            cropped_heatmap_gt, keypoints_2d_validity = self.generate_target(keypoints_2d_gt, keypoints_2d_mean, image_size, heatmap_size, cropped_heatmap.shape[-2:])
            keypoints_3d_validity = (keypoints_2d_validity.view(-1, n_views, keypoints_2d_validity.shape[-1]).sum(1) == n_views).float()
            # return htmap, cropped_heatmap_gt.view(-1, *cropped_heatmap_gt.shape[2:])

            if self.motion_guide:
                if keypoints_3d_history is not None:
                    keypoints_3d_history = keypoints_3d_history.view(batch_size // frames, frames-1, 17*3).transpose(1, 0).contiguous()
                    keypoints_3d_history += self.motion_disturb * torch.randn_like(keypoints_3d_history).cuda()
                else:
                    keypoints_2d_history = keypoints_2d_origin.view(batch_size // frames, frames, n_views, *keypoints_2d_origin.shape[-2:])
                    # keypoints_2d_history = keypoints_2d_history[:, :-1, :, :, :].contiguous()
                    alg_confidences_history = alg_confidences_history.view(batch_size // frames, frames, n_views, 17)
                    keypoints_2d_history = keypoints_2d_history.view(-1, n_views, 17, 2) * 4.0
                    alg_confidences_history = alg_confidences_history.view(-1, n_views, *alg_confidences.shape[1:])
                    alg_confidences_history = alg_confidences_history / alg_confidences_history.sum(dim=1, keepdim=True)
                    alg_confidences_history = alg_confidences_history + 1e-5  # for numerical stability
                    proj_matricies_history = proj_matricies.view(-1, frames, n_views, 3, 4)
                    proj_matricies_history = proj_matricies_history.view(-1, n_views, 3, 4)
                    keypoints_3d_history = multiview.triangulate_batch_of_points(
                        proj_matricies_history, keypoints_2d_history,
                        confidences_batch=alg_confidences_history
                    )
                    keypoints_3d_history = keypoints_3d_history * 1000.0
                    keypoints_3d_history = keypoints_3d_history.view(batch_size // frames, frames,
                                                                     17, 3).detach()

                smpl_refine = self.motion(keypoints_3d_history[:, :-1, :, :])
                keypoints_refine = batch_transform(smpl_refine[:, :, :-1, :].view(-1, 17, 3),
                                                   keypoints_3d_history[:, :-1, :, :].contiguous().view(
                                                       -1, 17, 3), parents=parents)
                pred = keypoints_refine.view(keypoints_3d_history.shape[0], -1, 17, 3) + smpl_refine[:, :, -1,
                                                                                               :].unsqueeze(2)
                # pred_3d_loss = torch.sqrt(((pred - keypoints_3d_history[:, 1:, :, :])**2).sum(-1)).mean()
                # print('3d: ', pred_3d_loss.item())
                # src = self.embedding(keypoints_3d_history.view(-1, 51)).view(frames-1, batch_size // frames, -1)
                # src = self.position(src)
                # if self.mask_subsequent:
                #     src_mask = self.motion.generate_square_subsequent_mask(frames - 1).cuda()
                #     tgt_mask = self.motion.generate_square_subsequent_mask(frames - 1).cuda()
                #     mem_mask = self.motion.generate_square_subsequent_mask(frames - 1).cuda()
                # else:
                #     src_mask = torch.zeros(frames - 1, frames - 1).cuda().float()
                #     tgt_mask = torch.zeros(frames - 1, frames - 1).cuda().float()
                #     mem_mask = torch.zeros(frames - 1, frames - 1).cuda().float()
                # pred = self.motion(src=src, tgt=src, src_mask=src_mask,
                #                    tgt_mask=tgt_mask, memory_mask=mem_mask)  # frames-1, batch, 256
                # pred = self.decoder(pred.view(-1, pred.shape[-1]))\
                #     .view(frames-1, batch_size // frames, -1)\
                #     .transpose(1, 0).contiguous()\
                #     .view(batch_size // frames, frames - 1, 17, 3)  # batch, frames-1, 17, 3
                proj_pred = proj_matricies.view(batch_size // frames, frames, n_views, 3, 4)[:, 1:, :, :].contiguous()  # batch, frames-1, 4, 3, 4
                pred = pred.unsqueeze(2).repeat(1, 1, n_views, 1, 1).view(-1, 3)
                proj_pred = proj_pred.unsqueeze(3).repeat(1, 1, 1, 17, 1, 1).view(-1, 3, 4)
                pred_2d = multiview.project_3d_points_to_image_batch(proj_pred, pred / 1000.0, convert_back_to_euclidean=True)
                pred_2d = pred_2d.view(-1, 17, 2)
                keypoints_2d_mean_history = keypoints_2d_mean.view(batch_size // frames, frames, n_views, 17, 2)[:, 1:, :, :, :].contiguous()
                keypoints_2d_mean_history = keypoints_2d_mean_history.view(-1, 17, 2)
                # pred_loss = torch.sqrt(((pred_2d.view(batch_size // frames, frames - 1, n_views, 17, 2) - keypoints_2d_gt.view(batch_size // frames, frames, n_views, 17, 2)[:, 1:, :, :, :]) ** 2).sum(-1)).mean()
                # print('2d:', pred_loss.item())
                cropped_heatmap_prior, _ = self.generate_target(pred_2d, keypoints_2d_mean_history,
                                                                                 image_size, heatmap_size,
                                                                                 cropped_heatmap.shape[-2:])
                cropped_heatmap_prior = cropped_heatmap_prior.view(batch_size // frames, frames - 1, n_views, 17, *cropped_heatmap_prior.shape[-2:])
                cropped_heatmap_pad = torch.ones_like(cropped_heatmap_prior[:, 0:1, :, :, :, :]).cuda()
                cropped_heatmap_prior = torch.cat([cropped_heatmap_pad, cropped_heatmap_prior], dim=1).view(-1, 17, *cropped_heatmap_prior.shape[-2:])
                cropped_heatmap = cropped_heatmap * 0.0 + cropped_heatmap * cropped_heatmap_prior * 1.0 + cropped_heatmap_prior * 0.0

            keypoints_2d_new, _ = op.integrate_tensor_2d(cropped_heatmap * self.heatmap_multiplier, self.heatmap_softmax)
            keypoints_2d_new = keypoints_2d_new.view(-1, n_views, *keypoints_2d_new.shape[-2:])
            keypoints_2d_new += keypoints_2d_mean.view(-1, n_views, *keypoints_2d_mean.shape[-2:]) - (torch.ones_like(keypoints_2d) * cropped_heatmap.shape[-1] // 2).cuda().float()
        else:
            keypoints_2d_new = keypoints_2d_origin.view(-1, n_views, *keypoints_2d_origin.shape[-2:])
            keypoints_2d_gt = keypoints_2d_gt.view(-1, *keypoints_2d_gt.shape[-2:])
            cropped_heatmap = heatmaps.clone()
            keypoints_2d_mean = torch.ones_like(keypoints_2d_gt, dtype=torch.long).cuda() * cropped_heatmap.shape[-1] // 2
            cropped_heatmap_gt, keypoints_2d_validity = self.generate_target(keypoints_2d_gt, keypoints_2d_mean, image_size, heatmap_size,
                                                    cropped_heatmap.shape[-2:])
            keypoints_3d_validity = (keypoints_2d_validity.view(-1, n_views, keypoints_2d_validity.shape[-1]).sum(
                1) == n_views).float()
            alg_confidences = alg_confidences_history

        # reshape back
        images = images.view(batch_size, n_views, *images.shape[1:])
        heatmaps = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])
        if alg_confidences is not None:
            alg_confidences = alg_confidences.view(batch_size, n_views, *alg_confidences.shape[1:])
            if vis is not None:
                alg_confidences = alg_confidences * vis
            # var_weight = multiview.get_heat_variance(cropped_heatmap.view(batch_size, n_views, *cropped_heatmap.shape[1:]))
            # alg_confidences = alg_confidences * var_weight * 4.0
            # norm confidences
            # alg_confidences = alg_confidences / torch.clamp_min(alg_confidences.sum(dim=1, keepdim=True), min=1e-5)
            alg_confidences = alg_confidences / alg_confidences.sum(dim=1, keepdim=True)
            alg_confidences = alg_confidences + 1e-5  # for numerical stability

        # calcualte shapes
        image_shape = tuple(images.shape[3:])
        batch_size, n_views, n_joints, heatmap_shape = heatmaps.shape[0], heatmaps.shape[1], heatmaps.shape[2], tuple(heatmaps.shape[3:])

        # change camera intrinsics
        new_K = torch.zeros_like(K)
        for batch_i in range(batch_size):
            for view_i in range(n_views):
                new_K[batch_i][view_i] = update_after_resize(K[batch_i][view_i], image_shape, heatmap_shape)

        proj_matricies_new = proj_matricies.float().cuda()

        # upscale keypoints_2d, because image shape != heatmap shape
        keypoints_2d_transformed = torch.zeros_like(keypoints_2d_new)
        keypoints_2d_transformed[:, :, :, 0] = keypoints_2d_new[:, :, :, 0] * (image_shape[1] / heatmap_shape[1])
        keypoints_2d_transformed[:, :, :, 1] = keypoints_2d_new[:, :, :, 1] * (image_shape[0] / heatmap_shape[0])
        # triangulate
        # print(proj_matricies.dtype, keypoints_2d_transformed.dtype)
        # print('proj_mat:', proj_matricies[0, 0, :, :])
        # print('kp2d:', keypoints_2d_transformed[0, 0, 0, :])
        try:
            if not svd:
                keypoints_2d_origin = keypoints_2d_origin.view(batch_size, n_views, *keypoints_2d_origin.shape[1:])
                keypoints_2d_transformed = 0.0 * keypoints_2d_transformed + keypoints_2d_origin * 4
                # print(alg_confidences)
            if self.agg_type != 'base' and self.warp:
                proj_matricies = torch.matmul(K_warp, proj_matricies.unsqueeze(2).repeat(1, 1, 17, 1, 1))
                keypoints_2d_transformed = torch.matmul(K_warp, torch.cat([keypoints_2d_transformed, torch.ones(batch_size, n_views, 17, 1).cuda()], dim=-1).unsqueeze(-1))
                keypoints_2d_transformed = keypoints_2d_transformed[:, :, :, :2, 0]
                keypoints_3d = multiview.triangulate_batch_of_points_warping(
                    proj_matricies, keypoints_2d_transformed,
                    confidences_batch=alg_confidences
                )
            else:

                keypoints_3d = multiview.triangulate_batch_of_points(
                    proj_matricies, keypoints_2d_transformed,
                    confidences_batch=alg_confidences
                )
                keypoints_3d_omits = []
                for i in range(n_views):
                    if i == 0:
                        idx = (1, 2, 3)
                    elif i == 1:
                        idx = (0, 2, 3)
                    elif i == 2:
                        idx = (0, 1, 3)
                    else:
                        idx = (0, 1, 2)
                    proj_matricies_omit = proj_matricies[:, idx, :, :]
                    keypoints_2d_transformed_omit = keypoints_2d_transformed[:, idx, :, :]
                    alg_confidences_omit = alg_confidences[:, idx, :]
                    # print(alg_confidences_omit.shape, keypoints_2d_transformed_omit.shape, proj_matricies_omit.shape)
                    keypoints_3d_omit = multiview.triangulate_batch_of_points(
                        proj_matricies_omit, keypoints_2d_transformed_omit,
                        confidences_batch=alg_confidences_omit
                    )
                    keypoints_3d_omits.append(keypoints_3d_omit * 1000.0)
                # for i in range(n_views):
                #     for j in range(i+1, n_views):
                #         idx = (i, j)
                #         proj_matricies_omit = proj_matricies[:, idx, :, :]
                #         keypoints_2d_transformed_omit = keypoints_2d_transformed[:, idx, :, :]
                #         alg_confidences_omit = alg_confidences[:, idx, :]
                #         # print(alg_confidences_omit.shape, keypoints_2d_transformed_omit.shape, proj_matricies_omit.shape)
                #         keypoints_3d_omit = multiview.triangulate_batch_of_points(
                #             proj_matricies_omit, keypoints_2d_transformed_omit,
                #             confidences_batch=alg_confidences_omit
                #         )
                #         keypoints_3d_omits.append(keypoints_3d_omit * 1000.0)
                keypoints_3d_omits = torch.stack(keypoints_3d_omits, dim=0)
            keypoints_3d = keypoints_3d * 1000.0


        except RuntimeError as e:
            print("Error: ", e)
            # print("proj_matricies = ", proj_matricies)
            # print("keypoints_2d_batch_pred =", keypoints_2d_transformed)
            exit()
        # print(cropped_heatmap_gt.shape)
        # cropped_heatmap_gt = cropped_heatmap_gt.view(-1, *cropped_heatmap_gt.shape[-2:])
        # cropped_heatmap = cropped_heatmap.view(-1, *cropped_heatmap.shape[-2:])
        if self.agg_type == 'base':
            return keypoints_3d_omits, keypoints_3d, keypoints_2d_new * 4.0, cropped_heatmap, proj_matricies_new, new_K, keypoints_2d_transformed, cropped_heatmap_gt, heatmaps, keypoints_2d_validity, keypoints_3d_validity
        else:
            return keypoints_3d_omits, keypoints_3d, keypoints_2d_new * 4.0, cropped_heatmap, proj_matricies_new, new_K, keypoints_2d_transformed, cropped_heatmap_gt, heatmaps, keypoints_2d_validity, keypoints_3d_validity

    def generate_target(self, keypoints_2d_gt, keypoints_2d_mean, image_size, heatmap_size, cropped_heatmap_size):
        '''
        :param joints:  [batch, num_joints, 2]
        :return: target
        '''

        feat_stride = (image_size / heatmap_size).cuda()
        mu_x = (keypoints_2d_gt[:, :, 0] / feat_stride[0] + 0.5).type(torch.long) - keypoints_2d_mean[:, :, 0] + cropped_heatmap_size[1] // 2
        mu_y = (keypoints_2d_gt[:, :, 1] / feat_stride[1] + 0.5).type(torch.long) - keypoints_2d_mean[:, :, 1] + cropped_heatmap_size[0] // 2
        keypoints_validity = (mu_x < cropped_heatmap_size[1]).float() * (mu_x >= 0).float() * (mu_y < cropped_heatmap_size[0]).float() * (mu_y >= 0).float()

        tmp_size = self.sigma * 3
        target = torch.zeros((keypoints_2d_gt.shape[0], keypoints_2d_gt.shape[1],
                           cropped_heatmap_size[0] + 2 * tmp_size,
                           cropped_heatmap_size[1] + 2 * tmp_size),
                          dtype=torch.float32).cuda()
        target = target.view(*target.shape[:2], -1)
        # Generate gaussian
        size = 2 * tmp_size + 1
        x = torch.arange(start=0, end=size, step=1, out=None).reshape(1, size)
        x = x.repeat(size, 1).cuda()
        y = x.transpose(0, 1)
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = torch.exp((-((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2)).float())
        g = g.unsqueeze(0).unsqueeze(0).repeat(keypoints_2d_gt.shape[0], keypoints_2d_gt.shape[1], 1, 1).view(keypoints_2d_gt.shape[0], keypoints_2d_gt.shape[1], -1)
        index_x_before = (
                    torch.arange(start=0, end=size, step=1).cuda().reshape(1, 1, size).repeat(keypoints_2d_gt.shape[0],
                                                                                              keypoints_2d_gt.shape[1],
                                                                                              1) + mu_x.unsqueeze(
                -1).repeat(1, 1, size)).unsqueeze(-2).repeat(1, 1, size, 1)
        index_x = torch.min(torch.max(torch.tensor(0), index_x_before.cpu()), torch.tensor(cropped_heatmap_size[1] + 2 * tmp_size - 1)).cuda()
        index_y_before = (
                    torch.arange(start=0, end=size, step=1).cuda().reshape(1, 1, size).repeat(keypoints_2d_gt.shape[0],
                                                                                              keypoints_2d_gt.shape[1],
                                                                                                    1) + mu_y.unsqueeze(
                -1).repeat(1, 1, size)).unsqueeze(-1).repeat(1, 1, 1, size)
        index_y = torch.min(torch.max(torch.tensor(0), index_y_before.cpu()), torch.tensor(cropped_heatmap_size[0] + 2 * tmp_size - 1)).cuda()
        index_y *= cropped_heatmap_size[1] + 2 * tmp_size
        index = (index_x + index_y).view(keypoints_2d_gt.shape[0], keypoints_2d_gt.shape[1], -1)
        # print(target.shape, index.shape, g.shape)
        target.scatter_(-1, index, g)
        target = target.view(keypoints_2d_gt.shape[0], keypoints_2d_gt.shape[1], cropped_heatmap_size[0] + 2*tmp_size, cropped_heatmap_size[1] + 2*tmp_size)
        target = target[:, :, tmp_size: -tmp_size, tmp_size: -tmp_size]

        return target, keypoints_validity

class VolumetricTriangulationNet(nn.Module):
    def __init__(self, config, device='cuda:0'):
        super().__init__()

        self.num_joints = config.model.backbone.num_joints
        self.volume_aggregation_method = config.model.volume_aggregation_method

        # volume
        self.volume_softmax = config.model.volume_softmax
        self.volume_multiplier = config.model.volume_multiplier
        self.volume_size = config.model.volume_size

        self.cuboid_side = config.model.cuboid_side

        self.kind = config.model.kind
        self.use_gt_pelvis = config.model.use_gt_pelvis

        # heatmap
        self.heatmap_softmax = config.model.heatmap_softmax
        self.heatmap_multiplier = config.model.heatmap_multiplier

        # transfer
        self.transfer_cmu_to_human36m = config.model.transfer_cmu_to_human36m if hasattr(config.model, "transfer_cmu_to_human36m") else False

        # modules
        config.model.backbone.alg_confidences = False
        config.model.backbone.vol_confidences = False
        if self.volume_aggregation_method.startswith('conf'):
            config.model.backbone.vol_confidences = True

        self.backbone = pose_resnet.get_pose_net(config.model.backbone, device=device)

        for p in self.backbone.final_layer.parameters():
            p.requires_grad = True

        self.process_features = nn.Sequential(
            nn.Conv2d(256, 32, 1)
        )

        self.volume_net = V2VModel(32, self.num_joints)

    def forward(self, images, K, proj_matricies, keypoints_3d):
        device = images.device
        batch_size, n_views = images.shape[:2]

        # reshape for backbone forward
        images = images.view(-1, *images.shape[2:])

        # forward backbone
        heatmaps, features, _, vol_confidences = self.backbone(images)

        # reshape back
        images = images.view(batch_size, n_views, *images.shape[1:])
        heatmaps = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])
        features = features.view(batch_size, n_views, *features.shape[1:])

        if vol_confidences is not None:
            vol_confidences = vol_confidences.view(batch_size, n_views, *vol_confidences.shape[1:])

        # calcualte shapes
        image_shape, heatmap_shape = tuple(images.shape[3:]), tuple(heatmaps.shape[3:])
        n_joints = heatmaps.shape[2]

        # norm vol confidences
        if self.volume_aggregation_method == 'conf_norm':
            vol_confidences = vol_confidences / vol_confidences.sum(dim=1, keepdim=True)

        # change camera intrinsics
        new_K = torch.zeros_like(K)
        for batch_i in range(batch_size):
            for view_i in range(n_views):
                new_K[batch_i][view_i] = update_after_resize(K[np.mod(batch_i, 20)][view_i], image_shape, heatmap_shape)

        proj_matricies_new = proj_matricies.float().to(device)

        # build coord volumes
        cuboids = []
        base_points = torch.zeros(batch_size, 3, device=device)
        coord_volumes = torch.zeros(batch_size, self.volume_size, self.volume_size, self.volume_size, 3, device=device)
        for batch_i in range(batch_size):
            # if self.use_precalculated_pelvis:
            if self.use_gt_pelvis:
                keypoints_3d_i = keypoints_3d[batch_i]
            else:
                keypoints_3d_i = keypoints_3d[batch_i]

            if self.kind == "coco":
                base_point = (keypoints_3d_i[11, :3] + keypoints_3d_i[12, :3]) / 2
            elif self.kind == "mpii":
                base_point = keypoints_3d_i[6, :3]

            base_points[batch_i] = base_point.to(device)

            # build cuboid
            sides = torch.from_numpy(np.array([self.cuboid_side,
                                               self.cuboid_side, self.cuboid_side])).float().to(device)
            position = base_point - sides / 2
            cuboid = volumetric.Cuboid3D(position, sides)
            cuboids.append(cuboid)

            # build coord volume
            xxx, yyy, zzz = torch.meshgrid(torch.arange(self.volume_size, device=device),
                                           torch.arange(self.volume_size, device=device),
                                           torch.arange(self.volume_size, device=device))
            grid = torch.stack([xxx, yyy, zzz], dim=-1).type(torch.float)
            grid = grid.reshape((-1, 3))

            grid_coord = torch.zeros_like(grid)
            grid_coord[:, 0] = position[0] + (sides[0] / (self.volume_size - 1)) * grid[:, 0]
            grid_coord[:, 1] = position[1] + (sides[1] / (self.volume_size - 1)) * grid[:, 1]
            grid_coord[:, 2] = position[2] + (sides[2] / (self.volume_size - 1)) * grid[:, 2]

            coord_volume = grid_coord.reshape(self.volume_size, self.volume_size, self.volume_size, 3)

            # random rotation
            if self.training:
                theta = np.random.uniform(0.0, 2 * np.pi)
            else:
                theta = 0.0

            if self.kind == "coco":
                axis = [0, 1, 0]  # y axis
            elif self.kind == "mpii":
                axis = [0, 0, 1]  # z axis

            center = base_point.type(torch.float).to(device)

            # rotate
            coord_volume = coord_volume - center
            coord_volume = volumetric.rotate_coord_volume(coord_volume, theta, axis)
            coord_volume = coord_volume + center

            # transfer
            if self.transfer_cmu_to_human36m:  # different world coordinates
                coord_volume = coord_volume.permute(0, 2, 1, 3)
                inv_idx = torch.arange(coord_volume.shape[1] - 1, -1, -1).long().to(device)
                coord_volume = coord_volume.index_select(1, inv_idx)

            coord_volumes[batch_i] = coord_volume

        # process features before unprojecting
        features = features.view(-1, *features.shape[2:])
        features = self.process_features(features)
        features = features.view(batch_size, n_views, *features.shape[1:])

        # lift to volume
        volumes = op.unproject_heatmaps(features, proj_matricies, coord_volumes, volume_aggregation_method=self.volume_aggregation_method, vol_confidences=vol_confidences)
        # integral 3d
        volumes = self.volume_net(volumes)
        vol_keypoints_3d, volumes = op.integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier, coord_volumes, softmax=self.volume_softmax)

        return vol_keypoints_3d, heatmaps, volumes, vol_confidences, cuboids, coord_volumes, base_points, new_K, proj_matricies_new


class VolumetricSelectedNet(nn.Module):
    def __init__(self, config, device='cuda:0'):
        super().__init__()

        self.num_joints = config.model.backbone.num_joints
        self.volume_aggregation_method = config.model.volume_aggregation_method

        # volume
        self.volume_softmax = config.model.volume_softmax
        self.volume_multiplier = config.model.volume_multiplier
        self.volume_size = config.model.volume_size

        self.cuboid_side = config.model.cuboid_side

        self.kind = config.model.kind
        self.use_gt_pelvis = config.model.use_gt_pelvis

        self.cons_view = config.model.cons_view
        self.thereshold = config.model.thereshold

        # heatmap
        self.heatmap_softmax = config.model.heatmap_softmax
        self.heatmap_multiplier = config.model.heatmap_multiplier

        # transfer
        self.transfer_cmu_to_human36m = config.model.transfer_cmu_to_human36m if hasattr(config.model, "transfer_cmu_to_human36m") else False

        # modules
        config.model.backbone.alg_confidences = False
        config.model.backbone.vol_confidences = False
        if self.volume_aggregation_method.startswith('conf'):
            config.model.backbone.vol_confidences = True

        self.backbone = pose_resnet.get_pose_net(config.model.backbone, device=device)

        for p in self.backbone.final_layer.parameters():
            p.requires_grad = False

        self.process_features = nn.Sequential(
            nn.Conv2d(256, 32, 1)
        )

        self.volume_net = V2VModel(32, self.num_joints)


    def forward(self, images, proj_matricies, batch):
        device = images.device
        batch_size, n_views = images.shape[:2]

        # reshape for backbone forward
        images = images.view(-1, *images.shape[2:])

        # forward backbone
        heatmaps, features, _, vol_confidences = self.backbone(images)

        # reshape back
        images = images.view(batch_size, n_views, *images.shape[1:])
        heatmaps = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])
        features = features.view(batch_size, n_views, *features.shape[1:])

        if vol_confidences is not None:
            vol_confidences = vol_confidences.view(batch_size, n_views, *vol_confidences.shape[1:])

        # calcualte shapes
        image_shape, heatmap_shape = tuple(images.shape[3:]), tuple(heatmaps.shape[3:])
        n_joints = heatmaps.shape[2]

        # norm vol confidences
        if self.volume_aggregation_method == 'conf_norm':
            vol_confidences = vol_confidences / vol_confidences.sum(dim=1, keepdim=True)

        # change camera intrinsics
        new_cameras = deepcopy(batch['cameras'])
        for view_i in range(n_views):
            for batch_i in range(batch_size):
                new_cameras[view_i][batch_i].update_after_resize(image_shape, heatmap_shape)

        proj_matricies = torch.stack([torch.stack([torch.from_numpy(camera.projection) for camera in camera_batch], dim=0) for camera_batch in new_cameras], dim=0).transpose(1, 0)  # shape (batch_size, n_views, 3, 4)
        proj_matricies = proj_matricies.float().to(device)

        # build coord volumes
        cuboids = []
        base_points = torch.zeros(batch_size, 3, device=device)
        coord_volumes = torch.zeros(batch_size, self.volume_size, self.volume_size, self.volume_size, 3, device=device)
        for batch_i in range(batch_size):
            # if self.use_precalculated_pelvis:
            if self.use_gt_pelvis:
                keypoints_3d = batch['keypoints_3d'][batch_i]
            else:
                keypoints_3d = batch['pred_keypoints_3d'][batch_i]

            if self.kind == "coco":
                base_point = (keypoints_3d[11, :3] + keypoints_3d[12, :3]) / 2
            elif self.kind == "mpii":
                base_point = keypoints_3d[6, :3]

            base_points[batch_i] = torch.from_numpy(base_point).to(device)

            # build cuboid
            sides = np.array([self.cuboid_side, self.cuboid_side, self.cuboid_side])
            position = base_point - sides / 2
            cuboid = volumetric.Cuboid3D(position, sides)

            cuboids.append(cuboid)

            # build coord volume
            xxx, yyy, zzz = torch.meshgrid(torch.arange(self.volume_size, device=device), torch.arange(self.volume_size, device=device), torch.arange(self.volume_size, device=device))
            grid = torch.stack([xxx, yyy, zzz], dim=-1).type(torch.float)
            grid = grid.reshape((-1, 3))

            grid_coord = torch.zeros_like(grid)
            grid_coord[:, 0] = position[0] + (sides[0] / (self.volume_size - 1)) * grid[:, 0]
            grid_coord[:, 1] = position[1] + (sides[1] / (self.volume_size - 1)) * grid[:, 1]
            grid_coord[:, 2] = position[2] + (sides[2] / (self.volume_size - 1)) * grid[:, 2]

            coord_volume = grid_coord.reshape(self.volume_size, self.volume_size, self.volume_size, 3)

            # random rotation
            if self.training:
                theta = np.random.uniform(0.0, 2 * np.pi)
            else:
                theta = 0.0

            if self.kind == "coco":
                axis = [0, 1, 0]  # y axis
            elif self.kind == "mpii":
                axis = [0, 0, 1]  # z axis

            center = torch.from_numpy(base_point).type(torch.float).to(device)

            # rotate
            coord_volume = coord_volume - center
            coord_volume = volumetric.rotate_coord_volume(coord_volume, theta, axis)
            coord_volume = coord_volume + center

            # transfer
            if self.transfer_cmu_to_human36m:  # different world coordinates
                coord_volume = coord_volume.permute(0, 2, 1, 3)
                inv_idx = torch.arange(coord_volume.shape[1] - 1, -1, -1).long().to(device)
                coord_volume = coord_volume.index_select(1, inv_idx)

            coord_volumes[batch_i] = coord_volume

        # process features before unprojecting
        features = features.view(-1, *features.shape[2:])
        features = self.process_features(features)
        features = features.view(batch_size, n_views, *features.shape[1:])

        features = torch.cat([features,heatmaps],2)


        # lift to volume
        #volumes = op.unproject_heatmaps(features, proj_matricies, coord_volumes, volume_aggregation_method=self.volume_aggregation_method, vol_confidences=vol_confidences)
        volumes = op.unproject_heatmaps_selected(features, proj_matricies, coord_volumes, self.cons_view,self.thereshold, volume_aggregation_method=self.volume_aggregation_method, vol_confidences=vol_confidences)
        # integral 3d
        volumes = self.volume_net(volumes)
        vol_keypoints_3d, volumes = op.integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier, coord_volumes, softmax=self.volume_softmax)

        return vol_keypoints_3d, heatmaps, volumes, vol_confidences, cuboids, coord_volumes, base_points


class VolumetricPointNet(nn.Module):
    def __init__(self, config, device='cuda:0'):
        super().__init__()

        self.num_joints = config.model.backbone.num_joints
        self.volume_aggregation_method = config.model.volume_aggregation_method

        # volume
        self.volume_softmax = config.model.volume_softmax
        self.volume_multiplier = config.model.volume_multiplier
        self.volume_size = config.model.volume_size

        self.cuboid_side = config.model.cuboid_side
        self.stage = config.model.stage
        self.kind = config.model.kind
        self.use_gt_pelvis = config.model.use_gt_pelvis

        # heatmap
        self.heatmap_softmax = config.model.heatmap_softmax
        self.heatmap_multiplier = config.model.heatmap_multiplier

        # transfer
        self.transfer_cmu_to_human36m = config.model.transfer_cmu_to_human36m if hasattr(config.model, "transfer_cmu_to_human36m") else False

        # modules
        config.model.backbone.alg_confidences = False
        config.model.backbone.vol_confidences = False
        if self.volume_aggregation_method.startswith('conf'):
            config.model.backbone.vol_confidences = True

        self.backbone = pose_resnet.get_pose_net(config.model.backbone, device=device)

        #for p in self.backbone.final_layer.parameters():
        #    p.requires_grad = False

        self.process_features = nn.Sequential(
            nn.Conv2d(256, 32, 1)
        )
        self.volume_net = V2VModel_point(32)
        self.point = PointNet(5000,32,self.num_joints,self.cuboid_side)

    def forward(self, images, proj_matricies, batch):
        device = images.device
        batch_size, n_views = images.shape[:2]

        # reshape for backbone forward
        images = images.view(-1, *images.shape[2:])

        # forward backbone
        heatmaps, features, _, vol_confidences = self.backbone(images)

        # reshape back
        images = images.view(batch_size, n_views, *images.shape[1:])
        heatmaps = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])
        features = features.view(batch_size, n_views, *features.shape[1:])

        if vol_confidences is not None:
            vol_confidences = vol_confidences.view(batch_size, n_views, *vol_confidences.shape[1:])

        # calcualte shapes
        image_shape, heatmap_shape = tuple(images.shape[3:]), tuple(heatmaps.shape[3:])
        n_joints = heatmaps.shape[2]

        # norm vol confidences
        if self.volume_aggregation_method == 'conf_norm':
            vol_confidences = vol_confidences / vol_confidences.sum(dim=1, keepdim=True)

        # change camera intrinsics
        new_cameras = deepcopy(batch['cameras'])
        for view_i in range(n_views):
            for batch_i in range(batch_size):
                new_cameras[view_i][batch_i].update_after_resize(image_shape, heatmap_shape)

        proj_matricies = torch.stack([torch.stack([torch.from_numpy(camera.projection) for camera in camera_batch], dim=0) for camera_batch in new_cameras], dim=0).transpose(1, 0)  # shape (batch_size, n_views, 3, 4)
        proj_matricies = proj_matricies.float().to(device)

        # build coord volumes
        cuboids = []
        base_points = torch.zeros(batch_size, 3, device=device)
        coord_volumes = torch.zeros(batch_size, self.volume_size, self.volume_size, self.volume_size, 3, device=device)
        for batch_i in range(batch_size):
            # if self.use_precalculated_pelvis:
            if self.use_gt_pelvis:
                keypoints_3d = batch['keypoints_3d'][batch_i]
            else:
                keypoints_3d = batch['pred_keypoints_3d'][batch_i]

            if self.kind == "coco":
                base_point = (keypoints_3d[11, :3] + keypoints_3d[12, :3]) / 2
            elif self.kind == "mpii":
                base_point = keypoints_3d[6, :3]

            base_points[batch_i] = torch.from_numpy(base_point).to(device)

            # build cuboid
            sides = np.array([self.cuboid_side, self.cuboid_side, self.cuboid_side])
            position = base_point - sides / 2
            cuboid = volumetric.Cuboid3D(position, sides)

            cuboids.append(cuboid)

            # build coord volume
            xxx, yyy, zzz = torch.meshgrid(torch.arange(self.volume_size, device=device), torch.arange(self.volume_size, device=device), torch.arange(self.volume_size, device=device))
            grid = torch.stack([xxx, yyy, zzz], dim=-1).type(torch.float)
            grid = grid.reshape((-1, 3))

            grid_coord = torch.zeros_like(grid)
            grid_coord[:, 0] = position[0] + (sides[0] / (self.volume_size - 1)) * grid[:, 0]
            grid_coord[:, 1] = position[1] + (sides[1] / (self.volume_size - 1)) * grid[:, 1]
            grid_coord[:, 2] = position[2] + (sides[2] / (self.volume_size - 1)) * grid[:, 2]

            coord_volume = grid_coord.reshape(self.volume_size, self.volume_size, self.volume_size, 3)

            # random rotation
            if self.training:
                theta = np.random.uniform(0.0, 2 * np.pi)
            else:
                theta = 0.0

            if self.kind == "coco":
                axis = [0, 1, 0]  # y axis
            elif self.kind == "mpii":
                axis = [0, 0, 1]  # z axis

            center = torch.from_numpy(base_point).type(torch.float).to(device)

            # rotate
            coord_volume = coord_volume - center
            coord_volume = volumetric.rotate_coord_volume(coord_volume, theta, axis)
            coord_volume = coord_volume + center

            # transfer
            if self.transfer_cmu_to_human36m:  # different world coordinates
                coord_volume = coord_volume.permute(0, 2, 1, 3)
                inv_idx = torch.arange(coord_volume.shape[1] - 1, -1, -1).long().to(device)
                coord_volume = coord_volume.index_select(1, inv_idx)

            coord_volumes[batch_i] = coord_volume

        # process features before unprojecting
        features = features.view(-1, *features.shape[2:])
        features = self.process_features(features)
        features = features.view(batch_size, n_views, *features.shape[1:])

        features = torch.cat([features,heatmaps],2)


        # lift to volume
        #volumes = op.unproject_heatmaps(features, proj_matricies, coord_volumes, volume_aggregation_method=self.volume_aggregation_method, vol_confidences=vol_confidences)
      
        volumes = op.unproject_heatmaps_selected3(features, proj_matricies, coord_volumes, volume_aggregation_method=self.volume_aggregation_method, vol_confidences=vol_confidences)
        # integral 3d
        B,F,_,_,_ = volumes.shape
        vol,_ = volumes.view(B,F,-1).sort(2,descending=True)
        #feature_volume = volumes[:,:-4,:,:,:] * (volumes.view(B,F,-1)[:,-1,:] > vol[:,-1,5000].view(B,1).repeat(1,64*64*64)).float().view(B,1,-1).repeat(1,F-4,1).view(B,F-4,64,64,64)
        feature_volume = self.volume_net(volumes[:,:-4,:,:,:])
        volumes = torch.cat([feature_volume,volumes[:,-4:,:,:,:]],1)
        vol_keypoints_3d = self.point(volumes)
        for i in range(self.stage-1):
            heatmaps_new = op.get_heatmaps(vol_keypoints_3d,proj_matricies,heatmaps.shape[3:])
            heatmaps = (heatmaps + heatmaps_new)/2.0
            att_volumes = op.unproject_heatmaps_selected4(heatmaps, proj_matricies, coord_volumes, volume_aggregation_method=self.volume_aggregation_method, vol_confidences=vol_confidences)
            #feature_volume = self.volume_net(volumes[:,:-4,:,:,:])volumes = torch.cat([feature_volume,volumes[:,-4:-1,:,:,:],att_volumes],1)
            #feature_volume = self.volume_net(volumes[:,:-4,:,:,:])
            volumes = torch.cat([volumes[:,:-1,:,:,:],att_volumes],1)
            vol_keypoints_3d = self.point(volumes)
        #vol_keypoints_3d, volumes = op.integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier, coord_volumes, softmax=self.volume_softmax)

        return vol_keypoints_3d, heatmaps, None, vol_confidences, cuboids, coord_volumes, base_points



class VolumetricFusionNet(nn.Module):
    def __init__(self, config, device='cuda:0'):
        super().__init__()

        self.num_joints = config.model.backbone.num_joints
        self.volume_aggregation_method = config.model.volume_aggregation_method

        # volume
        self.volume_softmax = config.model.volume_softmax
        self.volume_multiplier = config.model.volume_multiplier
        self.volume_size = config.model.volume_size

        self.cuboid_side = config.model.cuboid_side

        self.kind = config.model.kind
        self.use_gt_pelvis = config.model.use_gt_pelvis

        #self.cons_view = config.model.cons_view
        #self.thereshold = config.model.thereshold

        # heatmap
        self.heatmap_softmax = config.model.heatmap_softmax
        self.heatmap_multiplier = config.model.heatmap_multiplier

        # transfer
        self.transfer_cmu_to_human36m = config.model.transfer_cmu_to_human36m if hasattr(config.model, "transfer_cmu_to_human36m") else False

        # modules
        config.model.backbone.alg_confidences = False
        config.model.backbone.vol_confidences = False
        if self.volume_aggregation_method.startswith('conf'):
            config.model.backbone.vol_confidences = True

        self.backbone = pose_resnet.get_pose_net(config.model.backbone, device=device)

        self.IP = Inter_Perspective(32,self.num_joints)

        for p in self.backbone.final_layer.parameters():
            p.requires_grad = False
        #for p in self.IP.Mutual1.outconv.parameters():
        #    p.requires_grad = True
        #for p in self.IP.Mutual2.outconv.parameters():
        #    p.requires_grad = True

        self.process_features = nn.Sequential(
            nn.Conv2d(256, 32, 1)
        )

        self.volume_net = V2VModel(32, self.num_joints)


        #for p in self.volume_net.parameters():
        #    p.requires_grad = False


    def forward(self, images, proj_matricies, batch):
        device = images.device
        batch_size, n_views = images.shape[:2]

        # reshape for backbone forward
        images = images.view(-1, *images.shape[2:])

        # forward backbone
        heatmaps, features, _, vol_confidences = self.backbone(images)


        # reshape back
        images = images.view(batch_size, n_views, *images.shape[1:])
        heatmaps = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])
        features = features.view(batch_size, n_views, *features.shape[1:])

        if vol_confidences is not None:
            vol_confidences = vol_confidences.view(batch_size, n_views, *vol_confidences.shape[1:])

        # calcualte shapes
        image_shape, heatmap_shape = tuple(images.shape[3:]), tuple(heatmaps.shape[3:])

        # norm vol confidences
        if self.volume_aggregation_method == 'conf_norm':
            vol_confidences = vol_confidences / vol_confidences.sum(dim=1, keepdim=True)

        # change camera intrinsics
        new_cameras = deepcopy(batch['cameras'])
        for view_i in range(n_views):
            for batch_i in range(batch_size):
                new_cameras[view_i][batch_i].update_after_resize(image_shape, heatmap_shape)

        proj_matricies = torch.stack([torch.stack([torch.from_numpy(camera.projection) for camera in camera_batch], dim=0) for camera_batch in new_cameras], dim=0).transpose(1, 0)  # shape (batch_size, n_views, 3, 4)
        proj_matricies = proj_matricies.float().to(device)

        # build coord volumes
        cuboids = []
        base_points = torch.zeros(batch_size, 3, device=device)
        coord_volumes = torch.zeros(batch_size, self.volume_size, self.volume_size, self.volume_size, 3, device=device)
        for batch_i in range(batch_size):
            # if self.use_precalculated_pelvis:
            if self.use_gt_pelvis:
                keypoints_3d = batch['keypoints_3d'][batch_i]
            else:
                keypoints_3d = batch['pred_keypoints_3d'][batch_i]

            if self.kind == "coco":
                base_point = (keypoints_3d[11, :3] + keypoints_3d[12, :3]) / 2
            elif self.kind == "mpii":
                base_point = keypoints_3d[6, :3]

            base_points[batch_i] = torch.from_numpy(base_point).to(device)

            # build cuboid
            sides = np.array([self.cuboid_side, self.cuboid_side, self.cuboid_side])
            position = base_point - sides / 2
            cuboid = volumetric.Cuboid3D(position, sides)

            cuboids.append(cuboid)

            # build coord volume
            xxx, yyy, zzz = torch.meshgrid(torch.arange(self.volume_size, device=device), torch.arange(self.volume_size, device=device), torch.arange(self.volume_size, device=device))
            grid = torch.stack([xxx, yyy, zzz], dim=-1).type(torch.float)
            grid = grid.reshape((-1, 3))

            grid_coord = torch.zeros_like(grid)
            grid_coord[:, 0] = position[0] + (sides[0] / (self.volume_size - 1)) * grid[:, 0]
            grid_coord[:, 1] = position[1] + (sides[1] / (self.volume_size - 1)) * grid[:, 1]
            grid_coord[:, 2] = position[2] + (sides[2] / (self.volume_size - 1)) * grid[:, 2]

            coord_volume = grid_coord.reshape(self.volume_size, self.volume_size, self.volume_size, 3)

            # random rotation
            if self.training:
                theta = np.random.uniform(0.0, 2 * np.pi)
            else:
                theta = 0.0

            if self.kind == "coco":
                axis = [0, 1, 0]  # y axis
            elif self.kind == "mpii":
                axis = [0, 0, 1]  # z axis

            center = torch.from_numpy(base_point).type(torch.float).to(device)

            # rotate
            coord_volume = coord_volume - center
            coord_volume = volumetric.rotate_coord_volume(coord_volume, theta, axis)
            coord_volume = coord_volume + center

            # transfer
            if self.transfer_cmu_to_human36m:  # different world coordinates
                coord_volume = coord_volume.permute(0, 2, 1, 3)
                inv_idx = torch.arange(coord_volume.shape[1] - 1, -1, -1).long().to(device)
                coord_volume = coord_volume.index_select(1, inv_idx)

            coord_volumes[batch_i] = coord_volume

        # process features before unprojecting
        features = features.view(-1, *features.shape[2:])
        features = self.process_features(features)
        features, heatmaps = self.IP(features, n_views)
        heatmaps = heatmaps.view(2, batch_size, n_views, *heatmaps.shape[3:])
        features = features.view(batch_size, n_views, *features.shape[2:])

        #features = torch.cat([features,heatmaps],2)


        # lift to volume
        volumes = op.unproject_heatmaps(features, proj_matricies, coord_volumes, volume_aggregation_method=self.volume_aggregation_method, vol_confidences=vol_confidences)
        #volumes = op.unproject_heatmaps_selected(features, proj_matricies, coord_volumes, self.cons_view,self.thereshold, volume_aggregation_method=self.volume_aggregation_method, vol_confidences=vol_confidences)
        # integral 3d
        volumes = self.volume_net(volumes)
        vol_keypoints_3d, volumes = op.integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier, coord_volumes, softmax=self.volume_softmax)

        return vol_keypoints_3d, heatmaps, volumes, vol_confidences, cuboids, coord_volumes, base_points


class ResBlock(nn.Module):
    def __init__(self, f):
        super(ResBlock, self).__init__()
        self.f = f
        self.conv1 = nn.Conv2d(in_channels=self.f, out_channels=self.f, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.f, out_channels=self.f, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(self.f)
        self.bn2 = nn.BatchNorm2d(self.f)
        self.relu = nn.ReLU(True)
        # self.bn3 = nn.BatchNorm2d(num_features=self.f)

    def forward(self, x):
        y = self.relu(self.bn(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        x = self.relu(x + y)
        return x


class OffsetNet(nn.Module):
    def __init__(self,F):
        super(OffsetNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2*F,out_channels=F//2,kernel_size=3,stride=1,dilation=3,padding=3)
        self.off1  = nn.Conv2d(in_channels=F//2,out_channels=18,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(in_channels=2*F,out_channels=F//2,kernel_size=3,stride=1,dilation=6,padding=6)
        self.off2 = nn.Conv2d(in_channels=F // 2, out_channels=18, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=2*F,out_channels=F//2,kernel_size=3,stride=1,dilation=12,padding=12)
        self.off3 = nn.Conv2d(in_channels=F // 2, out_channels=18, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=2 * F, out_channels=F//2, kernel_size=3, stride=1, dilation=18, padding=18)
        self.off4 = nn.Conv2d(in_channels=F // 2, out_channels=18, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=2 * F, out_channels=F//2, kernel_size=3, stride=1, dilation=24, padding=24)
        self.off5 = nn.Conv2d(in_channels=F // 2, out_channels=18, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(True)
    def forward(self,x): ##x(1,2*F,H,W)->(1,18,H,W)
        off1 = self.off1(self.relu(self.conv1(x)))
        off2 = self.off2(self.relu(self.conv2(x)))
        off3 = self.off3(self.relu(self.conv3(x)))
        off4 = self.off4(self.relu(self.conv4(x)))
        off5 = self.off5(self.relu(self.conv5(x)))
        return [off1,off2,off3,off4,off5]


class Inter_Perspective(nn.Module): # B*4 pictures from 4 perspective
    def __init__(self,fnum,jnum):
        super(Inter_Perspective,self).__init__()
        self.features = fnum
        self.joints_num = jnum
        self.Mutual1 = MutualBlock(self.features) # return 2 feature maps(B,2*F,H,W) and 2 heatmaps(B,2*17,H,W)
        self.Mutual2 = MutualBlock(self.features)
        #self.Mutual3 = MutualBlock(self.features)
        self.offset = OffsetNet(self.features)
    def forward(self,x,n_views):
        BV,F,H,W = x.shape
        x = x.view(BV//n_views,n_views,F,H,W)
        image_pairs = []

        for i in range(n_views):
            for j in range(i+1,n_views):
                image_pairs.append(i)
                image_pairs.append(j)
        if n_views==4:
            feature_idx = (0,2,4,1,6,8,3,7,10,5,9,11)
        elif n_views ==3:
            feature_idx = (0,2,1,4,3,5)
        elif n_views ==2:
            feature_idx = (0,1)
        else:
            feature_idx = ()
        #feature_idx = tuple(feature_idx)
        image_pairs = tuple(image_pairs)
        pairs = len(image_pairs)//2
        f1 = x[:,image_pairs,:,:,:]
        f1 = torch.reshape(f1,(-1,2*F,H,W))
        off1 = self.offset(f1) ##5*(12,18,H,W)
        off1 = torch.cat(off1,1)
        f2,heatmaps1 = self.Mutual1(f1,off1)
        off2 = self.offset(f2) ##5*(12,18,H,W)
        off2 = torch.cat(off2,1)
        f3,heatmaps2 = self.Mutual2(f2,off2)
        #off3 = self.offset(f3) ##5*(12,18,H,W)
        #off3 = torch.cat(off3,1)
        #f4,heatmaps3 = self.Mutual3(f3,off3)
        num_joints = heatmaps1.shape[1]
        heatmaps1 = heatmaps1.view(BV // n_views, 2 * pairs, num_joints, H, W)
        heatmaps1 = heatmaps1[:,feature_idx,:,:,:]
        heatmaps1 = heatmaps1.view(BV // n_views, n_views, n_views - 1, num_joints, H, W)
        if self.training and False:
            if n_views >= 3:
                idx = random.randint(0,n_views-2)            
                heatmaps1 = (heatmaps1.sum(2) - heatmaps1[:,:,idx,:,:,:])/(n_views-2.0)
            else:
                heatmaps1 = heatmaps1.mean(2)
        else:
            heatmaps1 = heatmaps1.mean(2)
        heatmaps2 = heatmaps2.view(BV // n_views, 2 * pairs, num_joints, H, W)
        heatmaps2 = heatmaps2[:,feature_idx,:,:,:]
        heatmaps2 = heatmaps2.view(BV // n_views, n_views, n_views - 1, num_joints, H, W)
        if self.training and False:
            if n_views >= 3:
                idx = random.randint(0,n_views-2)
                heatmaps2 = (heatmaps2.sum(2) - heatmaps2[:,:,idx,:,:,:])/(n_views-2.0)
            else:
                heatmaps2 = heatmaps2.mean(2)
        else:
            heatmaps2 = heatmaps2.mean(2)
        features = f3.view(BV//n_views,pairs,2*F,H,W)
        features = features.view(BV//n_views,2*pairs,F,H,W)
        features = features[:,feature_idx,:,:,:]
        features = features.view(BV //n_views,n_views,n_views-1,F,H,W)
        if self.training and False:
            if n_views>=3:
                idx = random.randint(0,n_views-2)
                features = (features.sum(2)-features[:,:,idx,:,:,:])/(n_views-2.0)
            else:
                features = features.mean(2)
        else:
            features = features.mean(2)
        return features, torch.stack([heatmaps1,heatmaps2],0)


class MutualBlock(nn.Module):
    def __init__(self,F): #input:(6,F*2,H,W) 5*(6,18,H,W)
        super(MutualBlock,self).__init__()
        self.deform1 = DeformConv(in_channels=F,out_channels=F,kernel_size=3,dilation=3,padding=3)
        self.deform2 = DeformConv(in_channels=F, out_channels=F, kernel_size=3, dilation=6, padding=6)
        self.deform3 = DeformConv(in_channels=F, out_channels=F, kernel_size=3, dilation=12, padding=12)
        self.deform4 = DeformConv(in_channels=F, out_channels=F, kernel_size=3, dilation=18, padding=18)
        self.deform5 = DeformConv(in_channels=F, out_channels=F, kernel_size=3, dilation=24, padding=24)
        self.deform = [self.deform1,self.deform2,self.deform3,self.deform4,self.deform5]
        self.agg1 = ResBlock(F)
        self.agg2 = ResBlock(F)
        self.outconv = nn.Conv2d(in_channels=F,out_channels=17,kernel_size=3,padding=1,stride=1)
    def forward(self,features,offs):
        B,F,H,W = features.shape
        F = F//2
        #warped_feature = torch.zeros(B,F,H,W).cuda()
        warped_feature = []
        for i in range(5):
            warped_feature.append(self.deform[i](features[:,F:,:,:],offs[:,18*i:18*i+18,:,:]))
        warped_feature = torch.stack(warped_feature,0).sum(0)
        target_feature = features[:,:F,:,:]
        left = warped_feature+target_feature #(6,F,H,W)
        warped_feature = []
        for i in range(5):
            warped_feature.append(self.deform[i](features[:,:F,:,:],offs[:,18*i:18*i+18,:,:]))
        warped_feature = torch.stack(warped_feature,0).sum(0)
        target_feature = features[:,F:,:,:]
        right = warped_feature+target_feature #(6,F,H,W)
        leftright = torch.cat([left,right],0)
        out_features = self.agg2(self.agg1(leftright))
        #out_heat = self.outconv(leftright)
        out_heat = self.outconv(out_features)
        return torch.cat([out_features[:B,:,:,:],out_features[B:,:,:,:]],1),torch.reshape(torch.cat([out_heat[:B,:,:,:],out_heat[B:,:,:,:]],1),(-1,17,H,W))

