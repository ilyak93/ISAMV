

import cv2
import open3d as o3d
import numpy as np
from matplotlib.pyplot import imshow, show, hist
import os
from ctypes import wintypes, windll
from functools import cmp_to_key
from PIL import Image

import os
from torch.utils.data import Dataset
from torchvision.io import read_image

def winsort(data):
    _StrCmpLogicalW = windll.Shlwapi.StrCmpLogicalW
    _StrCmpLogicalW.argtypes = [wintypes.LPWSTR, wintypes.LPWSTR]
    _StrCmpLogicalW.restype  = wintypes.INT

    cmp_fnc = lambda psz1, psz2: _StrCmpLogicalW(psz1, psz2)
    return sorted(data, key=cmp_to_key(cmp_fnc))

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):

        all_files = os.listdir(img_dir)
        left_images = [img_dir+file for file in all_files if "left" in file]
        right_images = [img_dir+file for file in all_files if "right" in file]
        disparities = [img_dir+file for file in all_files if "dis" in file]
        depths = [img_dir+file for file in all_files if "depth" in file]
        self.left_images = winsort(left_images)
        self.right_image = winsort(right_images)
        self.disparities = winsort(disparities)
        self.depths = winsort(depths)
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
        return [left_image, right_image], [disparity, depth]