from time import sleep

import numpy as np
from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt
import cv2
from open3d.cpu.pybind.io import read_pinhole_camera_intrinsic
from ctypes import wintypes, windll
import os
from functools import cmp_to_key

def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1,3), colors])

    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property float red
        property float green
        property float blue
        end_header
    '''

    with open(filename, 'w') as f:
        f.write(ply_header % dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, '%f %f %f %f %f %f')



def winsort(data):
    _StrCmpLogicalW = windll.Shlwapi.StrCmpLogicalW
    _StrCmpLogicalW.argtypes = [wintypes.LPWSTR, wintypes.LPWSTR]
    _StrCmpLogicalW.restype  = wintypes.INT

    cmp_fnc = lambda psz1, psz2: _StrCmpLogicalW(psz1, psz2)
    return sorted(data, key=cmp_to_key(cmp_fnc))


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    #source_temp.paint_uniform_color([1, 0.706, 0])
    #target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

rs_ptc_path = "G:/Vista_project/finish_ipc/rs-ptc/"

rs_ptc_files = os.listdir(rs_ptc_path)
rs_ptc_files = winsort(rs_ptc_files)

st_ptc_path = "G:/Vista_project/finish_ipc/stereo-ptc/"

st_ptc_files = os.listdir(st_ptc_path)
st_ptc_files = winsort(st_ptc_files)

dest_path = "G:/Vista_project/finish_ipc/combined-ptc/"
if not os.path.exists(dest_path):
    os.mkdir(dest_path)

vis = o3d.visualization.Visualizer()
vis.create_window()

view_control = vis.get_view_control()
# view_control.rotate(180,180)
view_control.set_zoom(0.1)
front = np.array([1, 1, 1])
lookat = np.array([0, 0, 0])
up = np.array([0, 2, 0])
view_control.set_up(up)
view_control.set_lookat(lookat)
view_control.set_front(front)


for i in range(1, len(rs_ptc_files) + 1):
    pcd_rs = o3d.io.read_point_cloud("G:/Vista_project/finish_ipc/rs-ptc/"+ str(i) +"color.ply")
    pcd_stereo_therm = o3d.io.read_point_cloud("G:/Vista_project/finish_ipc/stereo-ptc/" + str(i) + ".ply")

    points_3D = np.asarray(pcd_stereo_therm.points)
    colors = np.asarray(pcd_stereo_therm.colors)
    temperature_filter = [False if (trio <= 0.2).sum() == 3 else True for trio in colors]
    temperature_filter_np = np.array(temperature_filter)
    mask = temperature_filter_np
    points_3D[:,0] = (points_3D[:, 0] + 0.1) / 0.9  # + = move right
    points_3D[:,1] = (points_3D[:, 1] + 0.1) / 1  # + = move down
    points_3D[:,2] = (points_3D[:, 2] - 2.25) / 1.5 # + = move forward

    pcd_stereo_therm.points = o3d.utility.Vector3dVector(points_3D[mask])
    pcd_stereo_therm.colors = o3d.utility.Vector3dVector(colors[mask])
    pcd_stereo_therm.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    #rotate = np.array([45.,0,0]).reshape([3,1])
    rot = np.asarray([[0.9743, -0.0197, -0.2241], [0, 0.9961, -0.0872], [0.2249, 0.0849, 0.9706]])
    center = np.array([0.,0.,0.])
    pcd_stereo_therm = pcd_stereo_therm.rotate(rot,center)

    #o3d.visualization.draw_geometries([pcd_rs, pcd_stereo_therm])
    #colors1 = np.asarray(pcd_rs.colors)
    #colors2 = np.asarray(pcd_stereo_therm.colors)

    pcd_rs.points.extend(pcd_stereo_therm.points)
    pcd_rs.colors.extend(pcd_stereo_therm.colors)

    points = np.asarray(pcd_rs.points)
    colors = np.asarray(pcd_rs.colors)

    output_file = dest_path + str(i) + ".ply"
    #create_output(points, colors, output_file)


    #if np.asarray(pcd.points).shape[0] > 1:
    #    less_then_indices = np.asarray(pcd.points)[:, 2] > -0.05
    #    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[less_then_indices])
    #    pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[less_then_indices])

    vis.add_geometry(pcd_rs)


    # view_control.set_up([1,1,1])

    vis.update_geometry(pcd_rs)
    vis.poll_events()
    vis.update_renderer()
    vis.clear_geometries()


