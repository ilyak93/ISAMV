import os
import numpy as np
import cv2
from PIL import Image

from ctypes import wintypes, windll
from functools import cmp_to_key
import shutil

def winsort(data):
    _StrCmpLogicalW = windll.Shlwapi.StrCmpLogicalW
    _StrCmpLogicalW.argtypes = [wintypes.LPWSTR, wintypes.LPWSTR]
    _StrCmpLogicalW.restype  = wintypes.INT

    cmp_fnc = lambda psz1, psz2: _StrCmpLogicalW(psz1, psz2)
    return sorted(data, key=cmp_to_key(cmp_fnc))


g_path = 'G:/Vista_project/finish_deep/calibration/single_camera_callibration/'
k = 0

dest_path = g_path + '/best_stereo200/'
thermal_left_dest_path = dest_path + '/left/'
thermal_right_dest_path = dest_path + '/right/'

if not os.path.exists(dest_path):
    os.mkdir(dest_path)
if not os.path.exists(thermal_left_dest_path):
    os.mkdir(thermal_left_dest_path)
if not os.path.exists(thermal_right_dest_path):
    os.mkdir(thermal_right_dest_path)

for i in range(6,14):
    thermal_left_source_path = g_path+str(i)+'/best_stereo200/left/'
    thermal_right_source_path = g_path + str(i) + '/best_stereo200/right/'

    tc_left_files = os.listdir(thermal_left_source_path)
    tc_left_files = winsort(tc_left_files)
    tc_right_files = os.listdir(thermal_right_source_path)
    tc_right_files = winsort(tc_right_files)

    for j in range(len(tc_left_files)):
        shutil.copy(thermal_left_source_path + tc_left_files[j], thermal_left_dest_path + str(k)+'.png')
        shutil.copy(thermal_right_source_path + tc_right_files[j], thermal_right_dest_path + str(k)+'.png')
        k = k + 1