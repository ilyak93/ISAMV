# skeleton source https://github.com/opencv/opencv/blob/master/samples/python/stereo_match.py
import os

import numpy as np
import cv2
import scipy
from matplotlib import pyplot as plt
import open3d as o3d
from scipy import io

from ctypes import wintypes, windll
from functools import cmp_to_key

def winsort(data):
    _StrCmpLogicalW = windll.Shlwapi.StrCmpLogicalW
    _StrCmpLogicalW.argtypes = [wintypes.LPWSTR, wintypes.LPWSTR]
    _StrCmpLogicalW.restype  = wintypes.INT

    cmp_fnc = lambda psz1, psz2: _StrCmpLogicalW(psz1, psz2)
    return sorted(data, key=cmp_to_key(cmp_fnc))

rs_ptc_path = "G:/Vista_project/finish1/stereo-therm-rs-depth-ptc/"

rs_ptc_files = os.listdir(rs_ptc_path)
rs_ptc_files = winsort(rs_ptc_files)

print('loading images...')
vis = o3d.visualization.Visualizer()
vis.create_window()

for i in range(len(rs_ptc_files)):
    ptc = o3d.io.read_point_cloud(rs_ptc_path+rs_ptc_files[i+50])
    #o3d.visualization.draw_geometries([ptc])

    vis.add_geometry(ptc)
    view_control = vis.get_view_control()
    #view_control.rotate(180,180)
    view_control.set_zoom(0.1)
    front = np.array([1, 1, 1])
    lookat = np.array([0, 0, 0])
    up = np.array([0, 2, 0])
    view_control.set_up(up)
    view_control.set_lookat(lookat)
    view_control.set_front(front)

    #view_control.set_up([1,1,1])

    vis.update_geometry(ptc)
    vis.poll_events()
    vis.update_renderer()
    vis.clear_geometries()
    #input("Press Enter to continue...")

