import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as T

from networks.deriveable_projection import PointCloudDropout, Projection
from networks.dp_utils import repeat_tensor_batch, quatmul, quatconj


def conv_block(in_ch, out_ch, ks=3, stride=1, padding=1, bn=False, act=True):
    "Basic convolutional block with relu and batchnorm"
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, ks, stride, padding, bias=not bn),
        nn.ReLU(True) if act else nn.Identity(),
        nn.BatchNorm2d(out_ch) if bn else nn.Identity(),
    )


def pose_predictor(n_ft):
    "Pose predictor with 2 hidden layers"
    return nn.Sequential(
        nn.Linear(n_ft, n_ft),
        nn.ReLU(True),
        nn.Linear(n_ft, n_ft),
        nn.ReLU(True),
        nn.Linear(n_ft, 4),
    )


def weight_init(m):
    "Kaiming normal init for relu with slope 0 for linear and conv layers"
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight.data, a=0)

'''
Encoder is almost as in original
paper, but with batch normalization and ReLU activations with kaiming normal initialization.
'''

class Encoder(nn.Module):
    "Encodes input images"

    def __init__(self, img_size, hidden_dim):
        super().__init__()
        self.convs = nn.Sequential(
            conv_block(3, 16, ks=5, stride=2, padding=2),
            conv_block(16, 16, ks=3, stride=2),
            conv_block(16, 16, ks=3, stride=1),
            conv_block(16, 16, ks=3, stride=2),
            conv_block(16, 16, ks=3, stride=1),
            conv_block(16, 16, ks=3, stride=2),
            conv_block(16, 16, ks=3, stride=1),
            conv_block(16, 16, ks=3, stride=2),
            conv_block(16, 16, ks=3, stride=1),
        )
        features_size = img_size // 8

        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(features_size ** 2, 1024, bias=True),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
        )

    def forward(self, img):
        conv_features = self.convs(img)
        features = self.features(conv_features)

        return features

'''
Pointcloud decoder used to generate point clouds from hidden vector of encoder
'''


class PointCloudDecoder(nn.Module):
    def __init__(self, num_points, hidden_dim=1024, predict_scale=True):
        super().__init__()
        self.num_points = num_points
        self.predict_scale = predict_scale
        self.pc_decoder = nn.Linear(hidden_dim, num_points * 3)
        self.scale_decoder = nn.Linear(hidden_dim, 1)

    def forward(self, z):
        "Transform hidden vector to pointcloud"
        # predict pointcloud
        pc = self.pc_decoder(z)
        pc = pc.view(-1, self.num_points, 3)
        pc = torch.tanh(pc) / 2.0

        scale = None
        if self.predict_scale:
            scale = self.scale_decoder(z)
            scale = torch.sigmoid(scale)

        return pc, scale

'''
Pose decoder In pose branch hidden_dim is slightly bigger than in paper. 
Pose decoder in eval mode returns one pose for each example in batch.
In train mode it returns num_candidates poses for each example
(one additional pose is student prediction).
In this case poses are repeated in batch dimension like num_candidates poses for first example,
..., num_candidates poses for nth example.
In this setting we can use repeat_tensor_batch to replicate point clouds
for each view and get projections.
'''

class PoseDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_candidates=4):
        "Predict `num_candidates` pose candidates for each example in batch"
        super().__init__()

        # Shared part
        self.ensemble = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
        )

        self.predictors = nn.ModuleList([pose_predictor(hidden_dim) for i in range(num_candidates)])

        self.student = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            pose_predictor(hidden_dim)
        )

    def forward(self, z):
        "Transform hidden vector to rotation quaternions"
        student_quat = self.student(z)

        if not self.training:
            return student_quat

        ensemble = self.ensemble(z)
        all_quats = [p(ensemble) for p in self.predictors]
        ensemble_quat = torch.cat(all_quats, dim=-1).view(-1, 4)

        return torch.cat([ensemble_quat, student_quat], dim=0)

'''
SupervisedModel uses camera supervision and only
predicts and projects pointclouds to different cameras
'''

class SupervisedModel(nn.Module):
    "Basic model"

    def __init__(
            self, img_size=128, hidden_dim=1024, num_points=8000,
            vox_size=64, smooth_sigma=3.0, predict_scale=True, keep_prob=0.07,
    ):
        super().__init__()
        self.encoder = Encoder(img_size, hidden_dim)
        self.decoder = PointCloudDecoder(num_points, hidden_dim, predict_scale)
        self.pc_dropout = PointCloudDropout(keep_prob)
        self.pc_projection = Projection(vox_size, smooth_sigma=smooth_sigma)

        self.encoder.apply(weight_init)
        self.decoder.apply(weight_init)

    def forward(self, imgs, poses):
        "Generate new view of `imgs` from `cameras` using differentiable projection"
        bs = imgs.size(0)
        num_views = poses.size(0) // bs

        z = self.encoder(imgs)
        pc, scale = self.decoder(z)

        pc = repeat_tensor_batch(self.pc_dropout(pc), num_views)
        scale = repeat_tensor_batch(scale, num_views)
        proj = self.pc_projection(pc, poses, scale)

        return proj


class SupervisedLoss(nn.Module):
    "Mse loss / 2"

    def forward(self, proj, masks, **kwargs):
        masks = F.interpolate(masks.unsqueeze(0), scale_factor=1 / 2, mode='bilinear', align_corners=True).squeeze()
        return dict(full_loss=F.mse_loss(proj, masks, reduction='sum') / (2 * proj.size(0)))



class UnsupervisedModel(nn.Module):
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