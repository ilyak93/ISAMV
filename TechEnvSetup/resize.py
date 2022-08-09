import os
import numpy as np
import cv2
from PIL import Image

from ctypes import wintypes, windll
from functools import cmp_to_key

def winsort(data):
    _StrCmpLogicalW = windll.Shlwapi.StrCmpLogicalW
    _StrCmpLogicalW.argtypes = [wintypes.LPWSTR, wintypes.LPWSTR]
    _StrCmpLogicalW.restype  = wintypes.INT

    cmp_fnc = lambda psz1, psz2: _StrCmpLogicalW(psz1, psz2)
    return sorted(data, key=cmp_to_key(cmp_fnc))

for t in range(3,5):
    g_path = 'G:/Vista_project/'
    path = g_path + str(t) + '/right/'
    dest_path = g_path + str(t) + '/right_resized/'
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)
    rs_width, rs_length = 1280, 720
    thermal_width, thermal_length = 640, 512
    tc_files = os.listdir(path)
    tc_files = winsort(tc_files)

    for i in range(len(tc_files)):
        tc1_np = cv2.imread(path + tc_files[i], cv2.IMREAD_UNCHANGED)
        resized = np.zeros((rs_length, rs_width), dtype=np.int)
        resized[0:thermal_length, 0:thermal_width] = tc1_np
        im = Image.fromarray(resized)
        im.save(dest_path + str(i) + "color.png", "PNG")