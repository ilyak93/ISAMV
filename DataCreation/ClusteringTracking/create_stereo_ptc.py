
import os
from time import sleep

import cv2
import numpy as np
import open3d as o3d
from ctypes import wintypes, windll
from functools import cmp_to_key
import matplotlib.pyplot as plt


def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1,3), colors])

    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
    '''

    with open(filename, 'w') as f:
        f.write(ply_header % dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, '%f %f %f %d %d %d')

def winsort(data):
    _StrCmpLogicalW = windll.Shlwapi.StrCmpLogicalW
    _StrCmpLogicalW.argtypes = [wintypes.LPWSTR, wintypes.LPWSTR]
    _StrCmpLogicalW.restype  = wintypes.INT

    cmp_fnc = lambda psz1, psz2: _StrCmpLogicalW(psz1, psz2)
    return sorted(data, key=cmp_to_key(cmp_fnc))

run_name = 'no_rect_no_mtlQ_pyDisparsity_ptc'
path = 'G:/Vista_project/finish_ipc/'
left_path = path + '/left/'
right_path = path + '/right/'
dest_path = path + run_name + '/'

#if not os.path.exists(dest_path):
#    os.mkdir(dest_path)

left_tc_files = os.listdir(left_path)
left_tc_files = winsort(left_tc_files)
right_tc_files = os.listdir(right_path)
right_tc_files = winsort(right_tc_files)

window_size = 10
min_disp = 1
num_disp = 64 - min_disp
stereo = cv2.StereoSGBM_create(
    minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = 10,
    P1 = 8 * 3 * window_size**2,
    P2 = 32 * 3 * window_size**2,
    disp12MaxDiff = 1,
    uniquenessRatio = 0,
    speckleWindowSize = 100,
    speckleRange = 3, mode=cv2.STEREO_SGBM_MODE_HH
    )

h, w = 512,640
focal_length = 0.8 * w

# Perspective transformation matrix
Q = np.float32([[1, 0, 0, -w / 2.0],
                [0, -1, 0, h / 2.0],
                [0, 0, 0, -focal_length],
                [0, 0, 1, 0]])

#vis = o3d.visualization.Visualizer()
#vis.create_window()

for i in range(50, len(left_tc_files)):
    tcL_np = cv2.imread("left2.png", cv2.IMREAD_UNCHANGED)
    tcR_np = cv2.imread("right2.png", cv2.IMREAD_UNCHANGED)
    imgL = np.round((tcL_np - tcL_np.min()) / (tcL_np.max() - tcL_np.min()) * 256).astype(np.uint8)
    imgR = np.round((tcR_np - tcR_np.min()) / (tcR_np.max() - tcR_np.min()) * 256).astype(np.uint8)

    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    points_3D = cv2.reprojectImageTo3D(disparity, Q)

    colors = cv2.cvtColor(imgL, cv2.COLOR_GRAY2RGB)
    mask_map = disparity > 1
    output_points = points_3D[mask_map]
    output_points[:,2] = output_points[:,2] #* 5
    output_colors = colors[mask_map]

    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(output_points)
    new_pcd.colors = o3d.utility.Vector3dVector(output_colors)

    output_file = "3dmap.ply"
    create_output(output_points, output_colors, output_file)

    print("Load a ply point cloud, print it, and render it")
    pcd_therm = o3d.io.read_point_cloud("./3dmap.ply")
    #pcd_therm.transform([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
    o3d.visualization.draw_geometries([pcd_therm])

    xyz_matrix = np.zeros((720, 1280,3), dtype=np.float64)
    xyz_matrix[:] = np.nan
    xyz_matrix[mask_map, :] = points_3D.reshape(720,1280,3)[mask_map, :]
    #indices = [[i,j] for i in range(mask_map.shape[0]) for j in range(mask_map.shape[1]) if mask_map[i,j] == True]
    #indices_np = np.asarray(indices)
    with open(dest_path+str(i)+'.npy', 'wb') as f:
        np.save(f, xyz_matrix)


    output_file = dest_path+str(i)+".ply"
    create_output(output_points, output_colors, output_file)



