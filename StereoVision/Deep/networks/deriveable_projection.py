import os
import time
import shutil
import math

from pathlib import Path
from collections import defaultdict

import numpy as np
import open3d as o3
import matplotlib.pyplot as plt

from PIL import Image
from scipy.io import loadmat
from scipy.spatial.transform import Rotation

import torch
import torch.nn as nn


from networks.dp_utils import pc_camera_transform, voxels_smooth, smoothing_kernel, drc_prob, pc_voxels, \
    repeat_tensor_batch


class Projection(nn.Module):
    "Diffefentiable point cloud projection module"
    def __init__(self, vox_size=64, smooth_ks=21, smooth_sigma=3.0):
        super().__init__()
        self.vox_size = vox_size
        self.ks = smooth_ks
        self.register_buffer('sigma', torch.tensor(smooth_sigma))

    def forward(self, pc, rotation, scale=None):
        "Project points `pc` to camera givne by `transform`"
        pc = pc_camera_transform(pc, rotation)
        voxels = pc_voxels(pc, self.vox_size)
        smooth = voxels_smooth(voxels, kernels=smoothing_kernel(self.sigma, self.ks), scale=scale)

        prob = drc_prob(smooth)
        proj = prob[:, :-1].sum(1).flip(1)
        return proj


class PointCloudDropout(nn.Module):
    "Drop random portions of pointclouds `pc`"
    def __init__(self, keep_prob=0.07):
        super().__init__()
        self.keep_prob = keep_prob

    def forward(self, pc):
        bs, n_points = pc.size(0), pc.size(1)
        n_keep = math.ceil(n_points * self.keep_prob)

        batch_idxs = repeat_tensor_batch(torch.arange(bs), n_keep)
        point_idxs = torch.cat([torch.randperm(n_points)[:n_keep] for i in range(bs)])

        return pc[batch_idxs, point_idxs].view(bs, n_keep, -1)



