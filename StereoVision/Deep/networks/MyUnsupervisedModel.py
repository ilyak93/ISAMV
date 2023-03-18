from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as T

from networks.DerivProjModels import Encoder, PoseDecoder, weight_init, PointCloudDecoder
from networks.deriveable_projection import PointCloudDropout, Projection
from networks.dp_utils import repeat_tensor_batch, quatmul, quatconj

def is_bn(m):
    return isinstance(m, nn.modules.batchnorm.BatchNorm2d) | isinstance(m, nn.modules.batchnorm.BatchNorm1d)

def take_bn_layers(model):
    for m in model.modules():
        if is_bn(m):
            yield m

class FreezedBnModel(nn.Module):
    def __init__(self, model, is_train=True):
        super(FreezedBnModel, self).__init__()
        self.model = model
        self.bn_layers = list(take_bn_layers(self.model))


    def forward(self, x):
        is_train = self.bn_layers[0].training
        if is_train:
            self.set_bn_train_status(is_train=False)
        predicted = self.model(x)
        if is_train:
            self.set_bn_train_status(is_train=True)

        return predicted

    def set_bn_train_status(self, is_train: bool):
        for layer in self.bn_layers:
            layer.train(mode=is_train)
            layer.weight.requires_grad = is_train
            layer.bias.requires_grad = is_train


class FreezedModel(nn.Module):
    def __init__(self, model):
        super(FreezedModel, self).__init__()
        self.model = model


    def forward(self, x):
        for param in self.model.parameters():
            param.requires_grad = False

        predicted = self.model(x)

        for param in self.model.parameters():
            param.requires_grad = True

        return predicted


class MyFilter(nn.Module):
    def __init__(self, init_indices, size=(720,1280)):
        self.filter = torch.zeros(size, requires_grad=False)
        self.filter = self.filter[init_indices] + 1
        self.filter.requires_grad_(True)

    def forward(self, imgs):
        return imgs * self.filter

class MyFilterRegLoss(nn.Module):
    def __init__(self, init_size):
        self.filter_size = init_size

    def forward(self, cur_filter):
        return self.filter_size - cur_filter.sum()

class MyUnsupervisedModelWrap(nn.Module):
    def __init__(self, myunsupervisedmodel):
        self.myunsupervisedmodel = myunsupervisedmodel
        self.myfrozensupervisedmodel = FreezedModel(
            FreezedBnModel(self.myunsupervisedmodel))
        self.is_frozen = False
        self.is_train = True

    def set_is_train(self, is_train):
        self.is_train = is_train

    def set_is_freezed(self, is_frozen):
        self.is_frozen = is_frozen

    def forward(self, imgs, poses):
        if self.is_train:
            if self.is_frozen:
                return self.myfrozensupervisedmodel(imgs, poses)
            else:
                return self.myunsupervisedmodel(imgs, poses)



class MyUnsupervisedModel(nn.Module):
    "Unsupervised model with ensemble of pose predictors"

    def __init__(self, img_size=128, vox_size=64,
                 z_dim=1024, pose_dim=128,
                 num_points=8000, num_candidates=4, num_views=5):
        super().__init__()
        self.num_views = num_views
        self.num_candidates = num_candidates

        self.encoder = Encoder(img_size, z_dim)
        self.pc_decoder = PointCloudDecoder(num_points, hidden_dim=z_dim)
        self.pc_dropout = PointCloudDropout()
        self.pc_projection = Projection(vox_size)
        self.pose_decoder = PoseDecoder(input_dim=z_dim, hidden_dim=pose_dim, num_candidates=num_candidates)

        self.encoder.apply(weight_init)
        self.pc_decoder.apply(weight_init)
        self.pose_decoder.apply(weight_init)

    def forward(self, imgs, poses):
        #imgs is only depth image
        #poses is the two thermal images
        z, z_p = self.encoder(imgs), self.encoder(poses)
        pc, scale = self.pc_decoder(z)
        poses = self.pose_decoder(z_p)

        # No ensemble on inference
        if not self.training:
            bs = imgs.size(0)
            pc = repeat_tensor_batch(self.pc_dropout(pc), self.num_views)
            scale = repeat_tensor_batch(scale, self.num_views)
            proj = self.pc_projection(pc, poses, scale)
            return proj, poses

        bs = imgs.size(0) * self.num_views
        ensemble_poses, student_poses = poses[:-bs], poses[-bs:]

        pc = repeat_tensor_batch(self.pc_dropout(pc), self.num_candidates * self.num_views)
        scale = repeat_tensor_batch(scale, self.num_candidates * self.num_views)
        proj = self.pc_projection(pc, ensemble_poses, scale)

        return proj, ensemble_poses, student_poses


class MySemiSupervisedModel(nn.Module):
    def __init__(self, dnet):
        self.dnet = dnet

        self.freezed_dnet = FreezedModel(FreezedBnModel(self.dnet))

class UnsupervisedLoss(nn.Module):
    "Loss combines projection losses for ensemble and student loss"

    def __init__(self, num_candidates=4, student_weight=20.0):
        super().__init__()
        self.student_weight = student_weight
        self.num_candidates = num_candidates

    def forward(self, pred, masks, training):
        proj, *poses = pred
        masks = F.interpolate(masks.unsqueeze(0), scale_factor=1 / 2, mode='bilinear', align_corners=True).squeeze()

        if not training:
            return dict(projection_loss=F.mse_loss(proj, masks, reduction='sum') / proj.size(0))

        ensemble_poses, student_poses = poses
        masks = repeat_tensor_batch(masks, self.num_candidates)

        projection_loss = F.mse_loss(proj, masks, reduction='none')
        projection_loss = projection_loss.sum((1, 2)).view(-1, self.num_candidates)
        min_idxs = projection_loss.argmin(dim=-1).detach()
        batch_idxs = torch.arange(min_idxs.size(0), device=min_idxs.device)

        # Student loss
        min_projection_loss = projection_loss[batch_idxs, min_idxs].sum() / min_idxs.size(0)
        ensemble_poses = ensemble_poses.view(-1, self.num_candidates, 4)
        best_poses = ensemble_poses[batch_idxs, min_idxs, :].detach()

        poses_diff = F.normalize(quatmul(best_poses, quatconj(student_poses)), dim=-1)
        angle_diff = poses_diff[:, 0]
        student_loss = (1 - angle_diff ** 2).sum() / min_idxs.size(0)

        # Save to print histogram
        self.min_idxs = min_idxs.detach()

        return dict(
            projection_loss=min_projection_loss,
            student_loss=student_loss,
            full_loss=min_projection_loss + self.student_weight * student_loss,
        )


class SemiSupervisedLoss(nn.Module):
    "Loss combines projection losses for ensemble and student loss"

    def __init__(self, num_candidates=4, student_weight=20.0):
        super().__init__()
        self.student_weight = student_weight
        self.num_candidates = num_candidates

    def forward(self, thermal_view1, thermal_view2, pred, SemiSupervisionNet, training):
        proj, *poses = pred

        #TODO: repeat_tensor_batch(masks, self.num_candidates) for thermal_view1/2
        projection_loss = SemiSupervisionNet(thermal_view1, thermal_view2, proj)

        if not training:
            return dict(projection_loss=projection_loss.sum(dim=0) / proj.size(0))

        ensemble_poses, student_poses = poses


        projection_loss = projection_loss.sum((1, 2)).view(-1, self.num_candidates)
        min_idxs = projection_loss.argmin(dim=-1).detach()
        batch_idxs = torch.arange(min_idxs.size(0), device=min_idxs.device)

        # Student loss
        min_projection_loss = projection_loss[batch_idxs, min_idxs].sum() / min_idxs.size(0)
        ensemble_poses = ensemble_poses.view(-1, self.num_candidates, 4)
        best_poses = ensemble_poses[batch_idxs, min_idxs, :].detach()

        poses_diff = F.normalize(quatmul(best_poses, quatconj(student_poses)), dim=-1)
        angle_diff = poses_diff[:, 0]
        student_loss = (1 - angle_diff ** 2).sum() / min_idxs.size(0)

        # Save to print histogram
        self.min_idxs = min_idxs.detach()

        return dict(
            projection_loss=min_projection_loss,
            student_loss=student_loss,
            full_loss=min_projection_loss + self.student_weight * student_loss,
        )


class model_to_train(Enum):
    dnet = 1
    projnet = 2
    filter = 3

class Network(nn.Module):

    def __init__(self, dnet, projnet, filter):
        self.dnet = dnet
        self.projnet = projnet
        self.filter = filter
        self.train_dpf = model_to_train.projnet

    def forward(self, x): #x[0] thermal1, x[1] thermal2, x[2] depth
        #filtered_depth = self.filter(x[2])
        proj = self.projnet(x) #filtered_proj = self.projnet(x[0], x[1], filtered_depth)
        pred_depth = self.dnet(x[0], x[1], proj)

        return pred_depth

    def set_model_to_train(self, model_to_train):
        self.train_dpf = model_to_train

        if self.train_dpf == model_to_train.dnet:
            self.dnet.set_is_freezed(True)
            self.projnet.set_is_freezed(False)
            self.filter.set_is_freezed(False)
        elif self.train_dpf == model_to_train.projnet:
            self.dnet.set_is_freezed(False)
            self.projnet.set_is_freezed(True)
            self.filter.set_is_freezed(False)
        else:
            self.dnet.set_is_freezed(False)
            self.projnet.set_is_freezed(False)
            self.filter.set_is_freezed(True)