import os
import time
import shutil
import math

from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.spatial.transform import Rotation

import torch

import torch.nn.functional as F
import torchvision.transforms.functional as T

from torchvision.utils import make_grid

def points2quat(v):
    "Convert xyz points to quaternions"
    assert len(v.shape) == 3
    assert v.size(-1) == 3
    return F.pad(v, (1, 0, 0, 0))


def quatmul(q1, q2):
    "Multiply quaternions"
    w1, x1, y1, z1 = torch.unbind(q1, dim=-1)
    w2, x2, y2, z2 = torch.unbind(q2, dim=-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return torch.stack([w, x, y, z], dim=-1)


def quatconj(q):
    "Conjugate of quaternion"
    m = q.new(4).fill_(-1)
    m[0] = 1.0
    return q * m


def quatrot(v, q, inverse=False):
    "Rotate points v [b, n, 3] with quaternions q [b, 4]"
    q = F.normalize(q, dim=-1)
    q = q[:, None, :]
    q_ = quatconj(q)
    v = points2quat(v)

    if inverse:
        wxyz = quatmul(quatmul(q_, v), q)
    else:
        wxyz = quatmul(quatmul(q, v), q_)

    if len(wxyz.shape) == 2:
        wxyz = wxyz.unsqueeze(0)

    return wxyz[:, :, 1:4]


def quat_from_campos(pos):
    "Convert blender camera format `pos` to torch tensor quaternion [w, x, y, z]"
    cx, cy, cz = pos[0]
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = cx / camDist
    cy = cy / camDist
    cz = cz / camDist
    t = math.sqrt(cx * cx + cy * cy)
    tx = cx / t
    ty = cy / t
    yaw = math.acos(tx)
    if ty > 0:
        yaw = 2 * math.pi - yaw

    roll = 0
    pitch = math.asin(cz)
    yaw = yaw + math.pi

    quat = Rotation.from_euler("yzx", [yaw, pitch, roll]).as_quat()
    quat = np.r_[quat[-1], quat[:-1]]

    return torch.tensor(quat.astype(np.float32))


def repeat_tensor_batch(tensor, times):
    "Repeat tensor `times` times for each element in batch"
    if tensor is None: return

    data_shape = tensor.shape[1:]
    repeats = [1, times] + [1] * len(data_shape)

    expanded = tensor.unsqueeze(1).repeat(*repeats)
    return expanded.view(-1, *data_shape)


def generate_projections_img(model, imgs, poses, masks):
    "Generate grid with model projections, gt projections and input images"
    device = next(model.parameters()).device
    proj, *_ = model(imgs[0].unsqueeze(0).to(device), poses.to(device))
    proj = proj.detach().cpu()

    grid = torch.cat([
        F.interpolate(imgs, scale_factor=1 / 2, mode='bilinear', align_corners=True),
        F.interpolate(masks.unsqueeze(1), scale_factor=1 / 2, mode='bilinear', align_corners=True).repeat(1, 3, 1, 1),
        proj.unsqueeze(1).repeat(1, 3, 1, 1),
    ])

    grid = make_grid(grid, nrow=imgs.size(0))
    return F.interpolate(grid.unsqueeze(0), scale_factor=2)


def pc_camera_transform(pc, rotation, focal_lenght=1.875, camera_distance=2.0):
    "Transform pontcloud `pc` to camera coordinates with `rotation`"

    pc = quatrot(pc, rotation)
    zs, ys, xs = torch.unbind(pc, dim=2)

    xs = xs * focal_lenght / (zs + camera_distance)
    ys = ys * focal_lenght / (zs + camera_distance)

    return torch.stack([zs, ys, xs], dim=2)


def pc_voxels(pc, size=64, eps=1e-6):
    "Create voxels of `[size]*3` from pointcloud `pc`"
    # save for later
    vox_size = pc.new(3).fill_(size)
    bs = pc.size(0)
    n = pc.size(1)

    # check borders
    valid = ((pc < 0.5 - eps) & (pc > -0.5 + eps)).all(dim=-1).view(-1)
    grid = (pc + 0.5) * (vox_size - 1)
    grid_floor = grid.floor()

    grid_idxs = grid_floor.long()
    batch_idxs = torch.arange(bs)[:, None, None].repeat(1, n, 1).to(pc.device)
    # idxs of form [batch, z, y, x] where z, y, x discretized indecies in voxel
    idxs = torch.cat([batch_idxs, grid_idxs], dim=-1).view(-1, 4)
    idxs = idxs[valid]

    # trilinear interpolation
    r = grid - grid_floor
    rr = [1. - r, r]
    voxels = []
    voxels_t = pc.new(bs, size, size, size).fill_(0)

    def trilinear_interp(pos):
        update = rr[pos[0]][..., 0] * rr[pos[1]][..., 1] * rr[pos[2]][..., 2]
        update = update.view(-1)[valid]

        shift_idxs = torch.LongTensor([[0] + pos]).to(pc.device)
        shift_idxs = shift_idxs.repeat(idxs.size(0), 1)
        update_idxs = idxs + shift_idxs
        valid_shift = update_idxs < size
        voxels_t.index_put_(torch.unbind(update_idxs, dim=1), update, accumulate=True)

        return voxels_t

    for k in range(2):
        for j in range(2):
            for i in range(2):
                voxels.append(trilinear_interp([k, j, i]))

    return torch.stack(voxels).sum(dim=0).clamp(0, 1)


def smoothing_kernel(sigma, kernel_size=21):
    "Generate 3 separate gaussian kernels with `sigma` stddev"
    x = torch.arange(-kernel_size//2 + 1., kernel_size//2 + 1., device=sigma.device)
    kernel_1d = torch.exp(-x**2 / (2. * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()

    k1 = kernel_1d.view(1, 1, 1, 1, -1)
    k2 = kernel_1d.view(1, 1, 1, -1, 1)
    k3 = kernel_1d.view(1, 1, -1, 1, 1)
    return [k1, k2, k3]

def voxels_smooth(voxels, kernels, scale=None):
    "Apply gaussian blur to voxels with separable `kernels` then `scale`"
    assert isinstance(kernels, list)

    # add fake channel for convs
    bs = voxels.size(0)
    voxels = voxels.unsqueeze(0)

    for k in kernels:
        # add padding for kernel dimension
        padding = [0] * 3
        padding[np.argmax(k.shape) - 2] = max(k.shape) // 2

        voxels = F.conv3d(voxels, k.repeat(bs, 1, 1, 1, 1), stride=1, padding=padding, groups=bs)

    voxels = voxels.squeeze(0)

    if scale is not None:
        voxels = voxels * scale.view(-1, 1, 1, 1)
        voxels = voxels.clamp(0, 1)

    return voxels


def drc_prob(voxels, clip_val=1e-5):
    "Compute termination probabilities from part 4 https://arxiv.org/pdf/1810.09381.pdf"
    inp = voxels.permute(1, 0, 2, 3)
    inp = inp.clamp(clip_val, 1.0 - clip_val)
    zero = voxels.new(1, inp.size(1), inp.size(2), inp.size(3)).fill_(clip_val)

    y = torch.log(inp)
    x = torch.log(1 - inp)

    r = torch.cumsum(x, dim=0)
    p1 = torch.cat([zero, r], dim=0)
    p2 = torch.cat([y, zero], dim=0)

    p = p1 + p2
    return torch.exp(p).permute(1, 0, 2, 3)
