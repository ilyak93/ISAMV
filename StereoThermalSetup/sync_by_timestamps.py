import os
import io

import numpy as np
import cv2
from PIL import Image


def find_closest_by_ts(tc_ts_files, rs_ts_files):
    closest_to_tc = [None] * len(tc_ts_files)

    for i in range(len(tc_ts_files),):
        dist = float('inf')
        for j in range(0, len(rs_ts_files), 2):
            tc_ts = int(tc_ts_files[i].split('_')[1].split('.')[0])
            rs_ts = int(rs_ts_files[j].split('_')[1].split('.')[0].split('c')[0])
            if abs(tc_ts - rs_ts) < dist:
                dist = abs(tc_ts - rs_ts)
                closest_to_tc[i] = j
    return closest_to_tc

from ctypes import wintypes, windll
from functools import cmp_to_key

def winsort(data):
    _StrCmpLogicalW = windll.Shlwapi.StrCmpLogicalW
    _StrCmpLogicalW.argtypes = [wintypes.LPWSTR, wintypes.LPWSTR]
    _StrCmpLogicalW.restype  = wintypes.INT

    cmp_fnc = lambda psz1, psz2: _StrCmpLogicalW(psz1, psz2)
    return sorted(data, key=cmp_to_key(cmp_fnc))

left_path = "G:/Vista_project/new_motion/left/"
right_path = "G:/Vista_project/new_motion/right/"
rs_path = "G:/Vista_project/new_motion/aligned_rs_both/"
dest_path = "G:/Vista_project/new_motion/sync/"
dest_presentation_path = "G:/Vista_project/new_motion/sync_presentation/"

left_tc_files = os.listdir(left_path)
left_tc_files = winsort(left_tc_files)
right_tc_files = os.listdir(right_path)
right_tc_files = winsort(right_tc_files)
rs_files = os.listdir(rs_path)
rs_files = winsort(rs_files)

closest_indices = find_closest_by_ts(left_tc_files, rs_files)
closest_indices2 = find_closest_by_ts(right_tc_files, rs_files)
tc_len = len(left_tc_files)
for i in range(tc_len):
    tc1_np = np.round(cv2.imread(left_path + left_tc_files[i], cv2.IMREAD_UNCHANGED) / 256).astype(np.uint8)
    tc2_np = np.round(cv2.imread(right_path + right_tc_files[i], cv2.IMREAD_UNCHANGED) / 256).astype(np.uint8)
    tc_combined = np.concatenate((tc1_np, tc2_np), axis=0)
    im = Image.fromarray(tc_combined)
    rs_bgr_img = cv2.imread(rs_path + rs_files[closest_indices[i]])
    im_rgb = cv2.cvtColor(rs_bgr_img, cv2.COLOR_BGR2RGB).astype(np.uint8)
    im_depth = np.round(cv2.imread(rs_path + rs_files[closest_indices[i]+1], cv2.IMREAD_UNCHANGED) / 256).astype(np.uint8)
    im_depth_3ch = np.stack((im_depth, im_depth, im_depth), axis=-1)
    rgb_depth_combined = np.concatenate((im_rgb, im_depth_3ch), axis=0 )
    all_combined = np.zeros((rgb_depth_combined.shape[0], rgb_depth_combined.shape[1] + tc_combined.shape[1], 3)).astype(np.uint8)
    all_combined[0:rgb_depth_combined.shape[0], 0:rgb_depth_combined.shape[1], :] = rgb_depth_combined
    all_combined[0:tc_combined.shape[0], rgb_depth_combined.shape[1]:, :] = np.stack((tc_combined, tc_combined, tc_combined), axis=-1)
    im = Image.fromarray(all_combined)
    im.save(dest_presentation_path + str(i) + ".png", "PNG")
    im = Image.fromarray(im_rgb)
    im.save(dest_path + str(i) + "color.png", "PNG")
    im = Image.fromarray(im_depth)
    im.save(dest_path + str(i) + "depth.png", "PNG")




