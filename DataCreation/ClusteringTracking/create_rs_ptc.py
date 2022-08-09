import numpy as np
import scipy
from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt
import cv2
from open3d.cpu.pybind.io import read_pinhole_camera_intrinsic
from ctypes import wintypes, windll
import os
from functools import cmp_to_key
from scipy.io import savemat


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

run_name = 'rs-ptc'
path = 'G:\Vista_project\objects-moving/1/'
color_path = path + '/aligned_rs/'
depth_path = path + '/aligned_rs/'
dest_path = path + run_name + '/'

if not os.path.exists(dest_path):
    os.mkdir(dest_path)

color_files = os.listdir(color_path)
color_files = [color_file for color_file in color_files if 'color' in color_file]
color_files = winsort(color_files)
depth_files = os.listdir(depth_path)
depth_files = [depth_file for depth_file in depth_files if 'depth' in depth_file]
depth_files = winsort(depth_files)

height = 720
width = 1280

import re

for i in range(100, len(color_files)):
    color_frame = color_files[i]
    depth_frame = depth_files[i]
    frame_name = color_frame.split('.')[0]
    frame_number = re.findall(r'\d+', frame_name)[0]
    np_color_im = cv2.imread(color_path+color_frame, cv2.IMREAD_COLOR)
    np_color_im = cv2.cvtColor(np_color_im, cv2.COLOR_BGR2RGB)
    np_depth_im = cv2.imread(depth_path+depth_frame, cv2.IMREAD_ANYDEPTH)
    #imgL = np.round((imgL_depth - imgL_depth.min()) / (imgL_depth.max() - imgL_depth.min()) * 256).astype(np.uint8)
    #imgR = np.round((imgR_depth - imgR_depth.min()) / (imgR_depth.max() - imgR_depth.min()) * 256).astype(np.uint8)



    color_raw = o3d.geometry.Image(np_color_im)
    depth_raw = o3d.geometry.Image(np_depth_im)

    '''
    depth_raw = o3d.geometry.Image(np.array(cv2.imread("0depth_aligned.png", cv2.IMREAD_ANYDEPTH)))
    plt.imshow(depth_raw)
    plt.show()
    dd = np.asarray(depth_raw)
    '''

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, depth_scale=1000.0, depth_trunc=1000, convert_rgb_to_intensity=False)
    '''
    print(rgbd_image)
    
    plt.subplot(1, 2, 1)
    plt.title('Redwood grayscale image')
    plt.imshow(rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('Redwood depth image')
    plt.imshow(rgbd_image.depth)
    plt.show()
    '''
    #intrinsic = read_pinhole_camera_intrinsic("real_sense_intrinsic")
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width=1280, height=720,
                                                  fx=1013.03991699219,
                                                  fy=1013.03991699219,
                                                  cx=655.809936523438,
                                                  cy=380.619934082031)

    '''
    depth intrinsics:
        Width:        1280
        Height:       720
        PPX:          655.809936523438
        PPY:          380.619934082031
        Fx:           1013.03991699219
        Fy:           1013.03991699219
    
    color intrinsics:
        Width:        1280
        Height:       720
        PPX:          648.483276367188
        PPY:          359.049194335938
        Fx:           787.998596191406
        Fy:           786.333374023438
    '''
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic)
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    #pcd.cluster_dbscan(eps=20, min_points=40, print_progress=True)
    o3d.visualization.draw_geometries([pcd])

    points_org = np.asarray(pcd.points)
    colors_org = (255*np.asarray(pcd.colors, dtype=float)).round().astype(np.uint8)
    #xmin = points_org.min(axis=0)
    #xmax = points_org.max(axis=0)
    #print()


    #output_file = dest_path+frame_name+".ply"
    #create_output(points_org, colors_org, output_file)

    new_points = []
    new_colors = []
    depthscale = 1000
    depth_image = np.asarray(rgbd_image.depth, dtype=np.float64) * depthscale
    color_image = np.asarray(rgbd_image.color)


    cx=655.809936523438
    cy=380.619934082031
    fx=1013.03991699219
    fy=1013.03991699219

    depthscale = 1000
    depth_image = np.asarray(rgbd_image.depth) * 1000
    U = np.asarray(list(range(width))).reshape(1, -1).repeat(720, axis=0)
    V = np.asarray(list(range(height))).reshape(-1, 1).repeat(1280, axis=1)
    Z = depth_image / depthscale
    X = (U - 655.809936523438) * Z / 1013.03991699219
    Y = (V - 380.619934082031) * Z / 1013.03991699219
    XYZ = np.stack((X, Y, Z), axis=-1)
    indices = depth_image > 0
    np_points = XYZ[indices, :]
    np_colors = color_image[indices, :]

    #new_pcd = o3d.geometry.PointCloud()
    #new_pcd.points = o3d.utility.Vector3dVector(np_points)
    #new_pcd.colors = o3d.utility.Vector3dVector(np_colors)
    #new_pcd.transform([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])

    #np_points = np.asarray(new_pcd.points)
    #pcd.cluster_dbscan(eps=20, min_points=40, print_progress=True)
    #o3d.visualization.draw_geometries([new_pcd])
    xyz_matrix = np.zeros((720, 1280, 3), dtype=np.float64)
    xyz_matrix[:] = np.nan
    xyz_matrix[indices, :] = XYZ[indices, :]
    #with open(dest_path + str(i) + '.npy', 'wb') as f:
    #    np.save(f, xyz_matrix)
    savemat(dest_path + frame_number + '.mat', {"xyz_matrix" : xyz_matrix})

    '''
    print("Load a ply point cloud created by matlab")
    pcd_rs = o3d.io.read_point_cloud("G:\Vista_project/finish1/rs-ptc/"+"ptc.ply")
    o3d.visualization.draw_geometries([pcd_rs])
    points = np.asarray(pcd_rs.points)
    mn = points.min(axis=0)
    mx = points.max(axis=0)
   '''





