import numpy as np
from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt
import cv2
from open3d.cpu.pybind.io import read_pinhole_camera_intrinsic
from ctypes import wintypes, windll
import os
from functools import cmp_to_key
from numpy.linalg import inv



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

rs_ptc_path = "G:/Vista_project/finish1/rs-ptc/"

rs_ptc_files = os.listdir(rs_ptc_path)
rs_ptc_files = [file for file in rs_ptc_files if 'ply' in file]
rs_ptc_files = winsort(rs_ptc_files)

st_ptc_path = "G:/Vista_project/finish1/stereo-ptc/"

st_ptc_files = os.listdir(st_ptc_path)
st_ptc_files = [file for file in st_ptc_files if 'ply' in file]
st_ptc_files = winsort(st_ptc_files)

dest_path = "G:/Vista_project/finish1/combined-ptc/"
if not os.path.exists(dest_path):
    os.mkdir(dest_path)
'''
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
'''

R_stereo_right_thermal_to_left_thermal = np.array([
    [0.999999587801456,	0.000459428446897368,	0.000783149040670592],
    [-0.000460719638714148,	0.999998533713736,	0.00164933556114786],
    [-0.000782390140674651,	-0.00164969569343721,	0.999998333183504]
])

t_stereo_right_thermal_to_left_thermal = np.array([-160.794024054281,	0.789913373933077,	-35.9582719836001]).reshape(-1,1)

R_stereo_left_thermal_to_right_rs = np.array([
    [0.999337308473166, 0.0130020206298207, -0.0339984022146633],
    [-0.0139396672988241, 0.999524904106065, -0.0274891205272156],
    [0.0336248356011803, 0.0279448301355261, 0.999043771263046]
])
t_stereo_left_thermal_to_right_rs = np.array([62.6154957188045, -91.7630998798625, 2.07628171113130]).reshape(-1,1)

M1 =  np.concatenate((R_stereo_right_thermal_to_left_thermal,
                      t_stereo_right_thermal_to_left_thermal),
                      axis=1)
M1 = np.concatenate((M1, np.array([0,0,0,1]).reshape(1,-1)), axis=0)
M1_inv = inv(M1)

M2 = np.concatenate((R_stereo_left_thermal_to_right_rs,
                      t_stereo_left_thermal_to_right_rs),
                      axis=1)
M2 = np.concatenate((M2, np.array([0,0,0,1]).reshape(1,-1)), axis=0)
transformation = M1_inv @ M2

for i in range(1, len(rs_ptc_files) + 1):
    pcd_rs = o3d.io.read_point_cloud("G:/Vista_project/finish1/rs-ptc/"+ str(i) +".ply")
    pcd_stereo_therm = o3d.io.read_point_cloud("G:/Vista_project/finish1/stereo-ptc/" + str(i) + ".ply")
    o3d.visualization.draw_geometries([pcd_rs])

    points_3D = np.asarray(pcd_stereo_therm.points)
    colors = np.asarray(pcd_stereo_therm.colors)
    color_filter = [False if (trio == [0, 0, 0]).sum() == 3 else True for trio in colors]
    color_filter_np = np.array(color_filter)
    mask = color_filter_np
    #points_3D[:, 0] = (points_3D[:, 0] + 0.1) / 0.9  # + = move right
    #points_3D[:, 1] = (points_3D[:, 1] + 0.1) / 1  # + = move down
    #points_3D[:, 2] = (points_3D[:, 2] - 2.25) / 1.5  # + = move forward

    pcd_stereo_therm.points = o3d.utility.Vector3dVector(points_3D[mask])
    pcd_stereo_therm.colors = o3d.utility.Vector3dVector(colors[mask])
    #pcd_stereo_therm.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # rotate = np.array([45.,0,0]).reshape([3,1])
    o3d.visualization.draw_geometries([pcd_stereo_therm])
    o3d.visualization.draw_geometries([pcd_rs, pcd_stereo_therm])

    pcd_stereo_therm.transform(M2)
    o3d.visualization.draw_geometries([pcd_rs, pcd_stereo_therm])
