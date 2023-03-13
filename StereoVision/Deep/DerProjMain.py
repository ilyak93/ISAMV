
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

import torch

import torchvision.transforms.functional as T

from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm

from networks.DerivProjModels import UnsupervisedModel, UnsupervisedLoss
from networks.dp_utils import quat_from_campos, generate_projections_img

'''
Data
We
use
splits and renders
from authors of

the
paper(but
we
train
on
train + test and validate
on
val).There is an
option in dataset
to
load
camera
positions
instead
of
other
views.We
always
load
all
5
renders(
for simplicity)

!wget - -quiet - -show - progress
"https://datasets.d2.mpi-inf.mpg.de/unsupervised-shape-pose/{DataBunch._ids["
chairs
"]}-renders.tar.gz"
!wget - -quiet - -show - progress
"https://datasets.d2.mpi-inf.mpg.de/unsupervised-shape-pose/{DataBunch._ids["
planes
"]}-renders.tar.gz"
!wget - -quiet - -show - progress
"https://datasets.d2.mpi-inf.mpg.de/unsupervised-shape-pose/{DataBunch._ids["
cars
"]}-renders.tar.gz"
02
958343 - renders.ta
100 % [ == == == == == == == == == = >] 434.78
M
4.95
MB / s in 1
m
46
s
!tar - xzf
"{DataBunch._ids["
chairs
"]}-renders.tar.gz"
!tar - xzf
"{DataBunch._ids["
planes
"]}-renders.tar.gz"
!tar - xzf
"{DataBunch._ids["
cars
"]}-renders.tar.gz"
!mv
"{DataBunch._ids["
chairs
"]}"
data
!mv
"{DataBunch._ids["
planes
"]}"
data
!mv
"{DataBunch._ids["
cars
"]}"
data

!rm
"{DataBunch._ids["
chairs
"]}-renders.tar.gz"
!rm
"{DataBunch._ids["
planes
"]}-renders.tar.gz"
!rm
"{DataBunch._ids["
cars
"]}-renders.tar.gz"
Pytorch
dataset

'''
def get_models(path=".", shapenet_id="03001627", split="train"):
    "Read model paths from split file"
    path = Path(path)

    assert split in ("train", "valid")
    split = path / f"{shapenet_id}.{split}"
    data = path / shapenet_id

    with open(split) as models:
        return [data / m.strip() for m in models]


class Shapenet(Dataset):
    "Dataset with renders and views for shapenet category"

    def __init__(self, models, camera=True, img_size=128):
        self.models = models
        self.camera = camera
        self.img_size = img_size

    def __getitem__(self, idx):
        model = self.models[idx]
        images = []
        masks = []
        cameras = []

        for name in sorted(os.listdir(model)):
            if name.startswith("render"):
                o = np.array(T.resize(Image.open(model / name), (self.img_size, self.img_size)))
                mask = o[..., -1].astype(np.float32) / 255.
                img = o[..., :-1].astype(np.float32) / 255.

                images.append(torch.tensor(img).permute(2, 0, 1))
                masks.append(torch.tensor(mask))

            if name.startswith("camera"):
                camera = loadmat(model / name)
                cameras.append(quat_from_campos(camera["pos"]))

        images = torch.stack(images)
        masks = torch.stack(masks)
        if self.camera:
            poses = torch.stack(cameras)
        else:
            poses = images

        return images, poses, masks

    def __len__(self):
        return len(self.models)

'''
Collate
function
handles
sampling
one
image
for point cloud generation.views might be cameras or images (depends on dataset configuration, but either way it is some information about other views)

'''
def multi_view_collate(batch):
    "Prepare batch with 1 image and n_poses masks and poses per item"
    bs = len(batch)
    n_poses = batch[0][0].size(0)

    idxs = torch.randint(0, n_poses, size=(bs,))
    imgs, poses, masks = zip(*[(img[i], view, mask) for (img, view, mask), i in zip(batch, idxs)])

    imgs = torch.stack(imgs)
    poses = torch.cat(poses, dim=0)
    masks = torch.cat(masks, dim=0)

    return imgs, poses, masks

'''
Datasets and dataloaders
all in one


class:
'''

class DataBunch():
    _ids = {
        "chairs": "03001627",
        "planes": "02691156",
        "cars": "02958343",
    }

    def __init__(self, path, category="chairs", batch_size=10, img_size=128, camera=True):
        train = get_models(path, self._ids[category], "train")
        valid = get_models(path, self._ids[category], "valid")
        self.train_ds, self.valid_ds = Shapenet(train, camera, img_size), Shapenet(valid, camera, img_size)
        self.train_dl = DataLoader(
            self.train_ds, batch_size,
            shuffle=True, collate_fn=multi_view_collate, drop_last=True,
            pin_memory=torch.cuda.is_available(), num_workers=4,
        )
        self.valid_dl = DataLoader(
            self.valid_ds, batch_size * 2,
            shuffle=False, collate_fn=multi_view_collate,
            pin_memory=torch.cuda.is_available(),
        )

'''
Training
Lets
try to overfit one example and see what happens

Overfit
one
batch
'''
data = DataBunch(path="data", batch_size=10, img_size=64, camera=False)
torch.random.manual_seed(10)
model = UnsupervisedModel(img_size=64, vox_size=32, num_points=2000).cuda()
loss = UnsupervisedLoss()

opt = torch.optim.Adam(model.parameters(), lr=1e-4)
torch.random.manual_seed(10)
imgs, poses, masks = next(iter(data.train_dl))

imgs = imgs.cuda()
poses = poses.cuda()
masks = masks.cuda()
pbar = tqdm(range(1000))

for i in pbar:
    adjust_params(model, i / 1000)
    t1 = time.perf_counter()
    proj = model(imgs, poses)

    l = loss(proj, masks, True)
    l["full_loss"].backward()
    opt.step()
    opt.zero_grad()
    dt = time.perf_counter() - t1

    pbar.set_postfix(time=dt, loss=l["full_loss"].item())

model.train()
proj, ensemble, student = model(imgs, poses)
batch = 2

for i in range(5):
    plt.figure(figsize=(10, 3))
    plt.subplot(151)
    plt.imshow(T.to_pil_image(poses[batch * 5 + i].cpu()));
    plt.axis(False)
    for j in range(4):
        plt.subplot(1, 5, j + 2)
        plt.title(20 * batch + 4 * i + j)
        plt.imshow(proj[20 * batch + 4 * i + j].detach().cpu(), cmap='gray')
        plt.axis(False)
plt.show()

plt.subplot(1, 5, j + 2)

plt.title(20 * batch + 4 * i + j)

plt.imshow(proj[20 * batch + 4 * i + j].detach().cpu(), cmap='gray')

plt.axis(False)

plt.show()

model.eval()
proj, student = model(imgs, poses)
i = 11
plt.subplot(121)
plt.imshow(T.to_pil_image(poses[i].detach().cpu()), cmap='gray')
plt.subplot(122)
plt.imshow(T.to_pil_image(proj[i].detach().cpu()), cmap='gray');
plt.show();

'''
Training
loop
'''

def loopy(dl):
    "Loop through dataloader indefinitley"
    while True:
        for o in dl: yield o


def adjust_params(model, step, keep_prob=(0.07, 1.0), sigma=(3.0, 0.2)):
    "Schedule model parameters linearly (dropout keep_prob and smoothing sigma)"
    assert 0 <= step <= 1

    new_keep_prob = keep_prob[0] * (1 - step) + keep_prob[1] * step
    new_sigma = sigma[0] * (1 - step) + sigma[1] * step

    model.pc_dropout.keep_prob = new_keep_prob
    model.pc_projection.sigma = torch.empty_like(model.pc_projection.sigma).fill_(new_sigma)

'''
Class
for training models and capturing progress
'''

class Learner:
    "Class for training a model"

    def __init__(self, path, data, model, loss, lr=1e-4, wd=0.001, seed=100):
        torch.random.manual_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.path = Path(path)
        if self.path.exists():
            shutil.rmtree(self.path)
        (self.path / "models").mkdir(exist_ok=True, parents=True)

        self.train_writer = SummaryWriter(log_dir=self.path / "logs" / "train")
        self.valid_writer = SummaryWriter(log_dir=self.path / "logs" / "valid")
        self.valid_losses = []

        self.data = data
        self.model = model.to(self.device)
        self.loss = loss
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)

    def fit(self, steps=300_000, eval_every=10_000, vis_every=1000,
            keep_prob=(0.07, 1.0), sigma=(3.0, 0.2), restore=None, start=None):
        "Train model for `steps` steps"
        start = 0

        if restore is not None:
            checkpoint = torch.load(restore, map_location="cpu")
            self.model.load_state_dict(checkpoint["model"])
            self.opt.load_state_dict(checkpoint["opt"])
            start = checkpoint["step"] if start is None else start

        self.pbar = tqdm(range(start + 1, steps + 1), desc="Step")
        train_dl = loopy(self.data.train_dl)

        for step in self.pbar:
            self.model.train()
            adjust_params(self.model, step / steps, keep_prob, sigma)

            self.step = step
            self.imgs, self.poses, self.masks = next(train_dl)
            self.one_batch()

            if step % eval_every == 0:
                self.model.eval()
                with torch.no_grad():
                    for self.imgs, self.poses, self.masks in tqdm(self.data.valid_dl, leave=False):
                        self.one_batch()
                    self.write_valid_losses()

                torch.save(
                    dict(model=self.model.state_dict(), opt=self.opt.state_dict(), step=self.step),
                    self.path / "models" / f"model_{self.step}.pth"
                )

            if step % vis_every == 0:
                self.model.eval()
                imgs, poses, masks = self.data.valid_ds[10]
                renders = generate_projections_img(self.model, imgs, poses, masks)
                self.train_writer.add_images("renders", renders, self.step)

    def one_batch(self):
        "Run one batch of a model"
        device = self.device
        imgs, poses, masks = self.imgs.to(device), self.poses.to(device), self.masks.to(device)
        proj = self.model(imgs, poses)
        t1 = time.perf_counter()
        loss = self.loss(proj, masks, training=self.model.training)
        if not self.model.training:
            self.valid_losses.append({key: l.item() for key, l in loss.items()})
            return

        loss['full_loss'].backward()
        self.opt.step()
        self.opt.zero_grad()
        dt = time.perf_counter() - t1
        self.pbar.set_postfix(time=dt, loss=loss['full_loss'].item())

        min_idxs = getattr(self.loss, "min_idxs", None)
        if min_idxs is not None:
            self.train_writer.add_histogram("other/predictors", min_idxs.cpu(), self.step)

        for key, l in loss.items():
            self.train_writer.add_scalar(key, l.item(), self.step)

    def write_valid_losses(self):
        "Calculate means for all validation losses and write to tensorboard"
        means = defaultdict(int)
        for loss in self.valid_losses:
            for key, val in loss.items():
                means[key] += val

        for key in means.keys():
            self.valid_writer.add_scalar(key, means[key] / len(self.valid_losses), self.step)
            print(f"{key}={means[key] / len(self.valid_losses):.3f}", end=" ")
        print()
        self.valid_losses = []

'''
Train
models
Chairs
'''

data = DataBunch(path="data", category="chairs", batch_size=24, camera=False)
learner = Learner(
    None,
    data,
    UnsupervisedModel(),
    UnsupervisedLoss(),
    lr=1e-3,
)
learner.fit(
    steps=130_000,
    eval_every=13_000,
    vis_every=2000,
)

#Planes

data = DataBunch(path="data", category="planes", img_size=64, batch_size=16, camera=False)
learner = Learner(
    None,
    data,
    UnsupervisedModel(img_size=64, vox_size=32, num_points=4000),
    UnsupervisedLoss(),
    lr=1e-4,
)
learner.fit(
    steps=30_000,
    start=0,
    eval_every=10_000,
    vis_every=1000,
    keep_prob=(0.256, 1.0),
    sigma=(2.44, 0.2),
    restore="./planes_unsupervised/models/model_80000.pth"
)
#Cars

data = DataBunch(path="data", category="cars", img_size=64, batch_size=16, camera=False)
learner = Learner(
    None,
    data,
    UnsupervisedModel(img_size=64, vox_size=32, num_points=4000),
    UnsupervisedLoss(),
    lr=1e-4,
)
learner.fit(
    steps=50_000,
    start=0,
    eval_every=10_000,
    vis_every=1000,
    keep_prob=(0.2095, 1.0),
    sigma=(2.58, 0.2),
    restore="./cars_unsupervised/models/model_60000.pth"
)

#Evaluation

font = {'fontname': 'DejaVu Serif', 'size': 12}

def eval_mode_figure(proj, poses, masks, category="chairs"):
    fig = plt.figure(figsize=(6, 10))
    for i in range(5):
        plt.subplot(5, 3, i * 3 + 1)
        if i == 0: plt.title("Image", **font)
        plt.imshow(T.to_pil_image(poses[i]))
        plt.axis(False)

        plt.subplot(5, 3, i * 3 + 2)
        if i == 0: plt.title("Mask", **font)
        plt.imshow(T.to_pil_image(masks.detach()[i]), cmap='gray')
        plt.axis(False)

        plt.subplot(5, 3, i * 3 + 3)
        if i == 0: plt.title("Student Projection", **font)
        plt.imshow(proj[i].detach(), cmap='gray')
        plt.axis(False)

    plt.show()
    fig.savefig(f"imgs/{category}_student.png", dpi=200, quality=100, bbox_inches="tight", pad_inches=0)


def train_mode_figure(proj, poses, masks, category="chairs"):
    fig = plt.figure(figsize=(10, 10))
    for i in range(5):
        plt.subplot(5, 5, i * 5 + 1)
        if i == 0: plt.title("Pose", **font)
        plt.imshow(T.to_pil_image(poses[i]))
        plt.axis(False)

        for j in range(4):
            plt.subplot(5, 5, i * 5 + j + 2)
            if i == 0: plt.title(f"Canditate {j + 1}", **font)
            plt.imshow(T.to_pil_image(proj[i * 4 + j]), cmap='gray')
            plt.axis(False)

    plt.show()
    fig.savefig(f"imgs/{category}_ensemble.png", dpi=200, quality=100, bbox_inches="tight", pad_inches=0)


#Chairs
#supervised

import pandas as pd

ours = pd.read_csv("chairs_supervised/ours_loss.csv")
auth = pd.read_csv("chairs_supervised/authors_loss.csv")
plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(8, 5))
plt.plot(ours.Step[::10], ours.Value[::10] / 2, label='Pytorch model', c="blue");
plt.plot(auth.Step[auth.Step <= ours.Step.max()], auth.Value[auth.Step <= ours.Step.max()], label='TF model',
         c="orange")
plt.ylim(0, 300)
plt.legend()
plt.xlabel('Step', **font)
plt.ylabel('MSE Loss', **font)
plt.show()
fig.savefig("plots/chairs_supervised_loss.png", bbox_inches="tight", pad_inches=0, dpi=200, quality=100)

#Chairs
#unsupervised

data = DataBunch(path="data", batch_size=2, camera=False)
model = UnsupervisedModel()

checkpoint = torch.load("chairs_unsupervised/models/model_91000.pth", map_location="cpu")
model.load_state_dict(checkpoint["model"])
adjust_params(model, 91 / 130)
imgs, poses, masks = data.train_ds[1]
model.eval();
proj, student = model(imgs[4].unsqueeze(0), poses)
proj = proj.cpu().detach()
eval_mode_figure(proj, poses, masks, "chairs")

model.train()
proj, ensemble, student = model(imgs[4].unsqueeze(0), poses)
proj = proj.detach()
train_mode_figure(proj, poses, masks, "chairs")

full_loss = pd.read_csv("chairs_unsupervised/train_full_loss.csv")
fig = plt.figure(figsize=(8, 5))
plt.plot(full_loss.Step, full_loss.Value, c="blue", alpha=0.4);
plt.plot(full_loss.Step, full_loss.Value.rolling(10).mean(), label="Full Loss", c="blue");
plt.ylabel("Unsupervised loss", **font)
plt.xlabel("Step", **font)
plt.xticks(**font)
plt.yticks(**font)
plt.legend(prop={"size": 12})
plt.show()
fig.savefig("plots/chairs_unsupervised_loss.png", bbox_inches="tight", pad_inches=0, dpi=200, quality=100)


#Point
#cloud
#vis

z = model.encoder(imgs[4].unsqueeze(0).cuda())
pc, scale = model.pc_decoder(z)
pc = model.pc_dropout(pc)
pc = pc.detach().cpu()
pco = o3.PointCloud()
pco.points = o3.Vector3dVector(pc[0].detach().numpy())
pco.transform([[0.7021877, -0.7118809, -0.0125779, 0],
               [-0.1033020, -0.0843846, -0.9910641, 0],
               [0.7044581, 0.6972122, -0.1327925, 0],
               [0, 0, 0, 1]])

vis = o3.JVisualizer()
vis.add_geometry(pco)
vis.show()

#Planes

data = DataBunch(path="data", category="planes", img_size=64, batch_size=2, camera=False)
model = UnsupervisedModel(img_size=64, vox_size=32, num_points=2000)

checkpoint = torch.load("planes_unsupervised/models/model_30000.pth", map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.pc_dropout.keep_prob = 1.0
imgs, poses, masks = data.train_ds[1002]
model.eval();
proj, student = model(imgs[3].unsqueeze(0), poses)
eval_mode_figure(proj, poses, masks, "planes")

model.train()
proj, ensemble, student = model(imgs[3].unsqueeze(0), poses)
train_mode_figure(proj, poses, masks, "planes")

full_loss = pd.read_csv("planes_unsupervised/full_loss.csv")
fig = plt.figure(figsize=(8, 5))
plt.plot(full_loss.Step, full_loss.Value, alpha=0.4, c="blue")
plt.plot(full_loss.Step, full_loss.Value.rolling(10).mean(), label="Full Loss", c="blue");
plt.ylabel("Unsupervised loss", **font)
plt.xlabel("Step", **font)
plt.xticks(**font)
plt.yticks(**font)
plt.legend(prop={"size": 12})
plt.show()
fig.savefig("plots/planes_unsupervised_loss.png", bbox_inches="tight", pad_inches=0, dpi=200, quality=100)

#Cars
data = DataBunch(path="data", category="cars", img_size=64, batch_size=2, camera=False)
model = UnsupervisedModel(img_size=64, vox_size=32, num_points=2000)

checkpoint = torch.load("cars_unsupervised/models/model_30000.pth", map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.pc_dropout.keep_prob = 1.0
imgs, poses, masks = data.train_ds[1002]
model.eval();
proj, student = model(imgs[3].unsqueeze(0), poses)
eval_mode_figure(proj, poses, masks, "cars")

model.train()
proj, ensemble, student = model(imgs[3].unsqueeze(0), poses)
train_mode_figure(proj, poses, masks, "cars")

full_loss = pd.read_csv("cars_unsupervised/full_loss.csv")
fig = plt.figure(figsize=(8, 5))
plt.plot(full_loss.Step, full_loss.Value, alpha=0.4, c="blue")
plt.plot(full_loss.Step, full_loss.Value.rolling(10).mean(), label="Full Loss", c="blue");
plt.ylabel("Unsupervised loss", **font)
plt.xlabel("Step", **font)
plt.xticks(**font)
plt.yticks(**font)
plt.legend(prop={"size": 12})
plt.show()
fig.savefig("plots/cars_unsupervised_loss.png", bbox_inches="tight", pad_inches=0, dpi=200, quality=100)

full_loss = pd.read_csv("bug/full_loss.csv")
fig = plt.figure(figsize=(8, 5))
plt.plot(full_loss.Step, full_loss.Value, alpha=0.4, c="blue")
plt.plot(full_loss.Step, full_loss.Value.rolling(10).mean(), label="Full Loss", c="blue");
plt.ylabel("Unsupervised loss", **font)
plt.xlabel("Step", **font)
plt.xticks(**font)
plt.yticks(**font)
plt.legend(prop={"size": 12})
plt.show()
fig.savefig("plots/bug_unsupervised_loss.png", bbox_inches="tight", pad_inches=0, dpi=200, quality=100)
