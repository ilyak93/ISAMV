

import cv2
import open3d as o3d
import numpy as np
from matplotlib.pyplot import imshow, show, hist
import os
import platform
if platform.system() == 'Windows':
    from ctypes import wintypes, windll
else:
    from natsort import natsorted
from functools import cmp_to_key
from PIL import Image
import random

import os
from torch.utils.data import Dataset
from torchvision.io import read_image

def winsort(data):
    _StrCmpLogicalW = windll.Shlwapi.StrCmpLogicalW
    _StrCmpLogicalW.argtypes = [wintypes.LPWSTR, wintypes.LPWSTR]
    _StrCmpLogicalW.restype  = wintypes.INT

    cmp_fnc = lambda psz1, psz2: _StrCmpLogicalW(psz1, psz2)
    return sorted(data, key=cmp_to_key(cmp_fnc))

sort_func = winsort if platform.system() == 'Windows' else natsorted 

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):

        all_files = os.listdir(img_dir)
        left_images = [img_dir+file for file in all_files if "left" in file]
        right_images = [img_dir+file for file in all_files if "right" in file]
        disparities = [img_dir+file for file in all_files if "dis" in file]
        depths = [img_dir+file for file in all_files if "depth" in file]
        self.left_images = sort_func(left_images)
        self.right_image = sort_func(right_images)
        self.disparities = sort_func(disparities)
        self.depths = sort_func(depths)
        self.transform = transform
        self.size = len(self.left_images)
        self.target_transform = target_transform

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        left_image = cv2.imread(self.left_images[idx], cv2.IMREAD_ANYDEPTH)
        left_image = np.array(left_image, dtype='float32')
        right_image = cv2.imread(self.right_image[idx], cv2.IMREAD_ANYDEPTH)
        right_image = np.array(right_image, dtype='float32')
        disparity = cv2.imread(self.disparities[idx], cv2.IMREAD_ANYDEPTH)
        disparity = np.array(disparity, dtype='float32')
        depth = cv2.imread(self.depths[idx], cv2.IMREAD_ANYDEPTH)
        depth = np.array(depth, dtype='float32')

        #if self.transform:
        #    left_image = self.transform(left_image)
        #    right_image = self.transform(right_image)
        #if self.target_transform:
        #    label = self.target_transform(label)
        if self.transform:
            if random.random() < 0.5:
                left_image, right_image, disparity, depth = \
                np.flipud(left_image), np.flipud(right_image), np.flipud(disparity), np.flipud(depth)
            if random.random() < 0.5:
                left_image, right_image, disparity, depth = \
                np.fliplr(left_image), np.fliplr(right_image), np.fliplr(disparity), np.fliplr(depth)
            if random.random() < 0.5:
                mu, sigma = 0, 2000 # mean and standard deviation
                r = np.random.normal(mu, sigma, left_image.shape)
                left_image[left_image + r > 2 ** 16 - 1] = 2 ** 16 - 1
                left_image[left_image + r <= 2 ** 16 - 1] = left_image[left_image + r <= 2 ** 16 - 1] + r 
            if random.random() < 0.5:
                mu, sigma = 0, 2000 # mean and standard deviation
                r = np.random.normal(mu, sigma, left_image.shape)
                right_image[right_image + r > 2 ** 16 - 1] = 2 ** 16 - 1
                right_image[right_image + r <= 2 ** 16 - 1] = right_image[right_image + r <= 2 ** 16 - 1] + r
            if random.random() < 0.5:
                r = random.random() * 2000
                left_image[left_image + r > 2 ** 16 - 1] = 2 ** 16 - 1
                left_image[left_image + r <= 2 ** 16 - 1] = left_image[left_image + r <= 2 ** 16 - 1] + r
            if random.random() < 0.5:
                r = random.random() * 2000
                right_image[right_image + r > 2 ** 16 - 1] = 2 ** 16 - 1
                right_image[right_image + r <= 2 ** 16 - 1] = right_image[right_image + r <= 2 ** 16 - 1] + r
        return [left_image.copy(), right_image.copy()], [disparity.copy(), depth.copy()]