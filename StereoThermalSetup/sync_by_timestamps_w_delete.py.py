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

def my_find_closest_by_ts(tc1_ts_files, tc2_ts_files):
    closest_to_tc = [None] * len(tc1_ts_files)

    for i in range(len(tc1_ts_files),):
        dist = float('inf')
        for j in range(len(tc2_ts_files)):
            tc1_ts = int(tc1_ts_files[i].split('_')[1].split('.')[0])
            tc2_ts = int(tc2_ts_files[j].split('_')[1].split('.')[0])
            if abs(tc1_ts - tc2_ts) < dist:
                dist = abs(tc1_ts - tc2_ts)
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




for g in range(21,22):
    g_path = "F:/Vista_project2/" + str(g) + "_ready"
    left_path = g_path + "/left/"
    right_path = g_path + "/right/"
    rs_path = g_path + "/rs/"
    dest_path = g_path + "/sync/"
    dest_presentation_path = g_path + "/sync_presentation/"
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)
    if not os.path.exists(dest_presentation_path):
        os.mkdir(dest_presentation_path)

    left_tc_files = os.listdir(left_path)
    left_tc_files = winsort(left_tc_files)
    right_tc_files = os.listdir(right_path)
    right_tc_files = winsort(right_tc_files)
    rs_files = os.listdir(rs_path)
    rs_files = winsort(rs_files)


    #ci = my_find_closest_by_ts(left_tc_files, right_tc_files)
    closest_indices = find_closest_by_ts(left_tc_files, rs_files)
    valid_rs_files = [rs_files[i] for i in closest_indices]
    files_to_delete = [file for file in rs_files if file not in valid_rs_files and "color" in file]
    #closest_indices2 = find_closest_by_ts(right_tc_files, rs_files)

    for file_to_delete in files_to_delete:
        file_name = file_to_delete[:-9]
        color_file = file_to_delete
        depth_file = file_name + "depth.png"
        os.remove(rs_path+color_file)
        os.remove(rs_path + depth_file)

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
        im_depth = cv2.imread(rs_path + rs_files[closest_indices[i] + 1], cv2.IMREAD_UNCHANGED)
        im = Image.fromarray(im_depth)
        im.save(dest_path + str(i) + "depth.png", "PNG")




