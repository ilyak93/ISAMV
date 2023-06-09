# two approaches: A. to project depth directly onto  left and right B. to deproject and project onto left and right
# also with tangential distortion cooefs and without

# first approach:
import cv2
import open3d as o3d
import numpy as np
from matplotlib.pyplot import imshow, show
import os
from ctypes import wintypes, windll
from functools import cmp_to_key
from PIL import Image


def winsort(data):
    _StrCmpLogicalW = windll.Shlwapi.StrCmpLogicalW
    _StrCmpLogicalW.argtypes = [wintypes.LPWSTR, wintypes.LPWSTR]
    _StrCmpLogicalW.restype = wintypes.INT

    cmp_fnc = lambda psz1, psz2: _StrCmpLogicalW(psz1, psz2)
    return sorted(data, key=cmp_to_key(cmp_fnc))


# 454 - 700
# np_color_im = cv2.imread("G:/Vista_project/finish_deep/aligned_rs/650color.png", cv2.IMREAD_COLOR)
# np_color_im = cv2.cvtColor(np_color_im, cv2.COLOR_BGR2RGB)
# np_depth_im = cv2.imread("G:/Vista_project/finish_deep/aligned_rs/650depth.png", cv2.IMREAD_ANYDEPTH)
# np_left_im = cv2.imread("G:/Vista_project/finish_deep/left_resized/650color.png", cv2.IMREAD_ANYDEPTH)

quartet_folder_name = "sync"

dataset_num = 21
gen_path = "F:/Vista_project2/"
all_path = gen_path + str(dataset_num) + "_ready/" + quartet_folder_name + "/"
all_files = os.listdir(all_path)
all_files = winsort(all_files)

# file num 62
# file num 726
# TODO: there are still not synced frames (singles)
# TODO: bad sync at fast rotations or fast velocity/accelerations
file_num = 3339
files = 4
# file_num = file_num - 1
thermal_file_path = gen_path + str(dataset_num) + "_ready/" + \
                    quartet_folder_name + "/" + \
                    all_files[file_num * files + 2]
if "right" in thermal_file_path:
    right_file_path = thermal_file_path
    left_file_path = gen_path + str(dataset_num) + "_ready/" + \
                     quartet_folder_name + "/" + all_files[file_num * files + 3]
else:
    left_file_path = thermal_file_path
    right_file_path = gen_path + str(dataset_num) + "_ready/" + \
                      quartet_folder_name + "/" + all_files[file_num * files + 3]

color_file_path = gen_path + str(dataset_num) + \
                  "_ready/" + quartet_folder_name + "/" + \
                  all_files[file_num * files]
depth_file_path = gen_path + str(dataset_num) + "_ready/" + \
                  quartet_folder_name + "/" + all_files[file_num * files + 1]

np_color_im = cv2.imread(color_file_path, cv2.IMREAD_COLOR)
np_color_im = cv2.cvtColor(np_color_im, cv2.COLOR_BGR2RGB)
np_depth_im = cv2.imread(depth_file_path, cv2.IMREAD_ANYDEPTH)
np_left_im = cv2.imread(left_file_path, cv2.IMREAD_ANYDEPTH)

color_raw = o3d.geometry.Image(np_color_im)
depth_raw = o3d.geometry.Image(np_depth_im)

rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw, depth_raw, depth_scale=1000.0, depth_trunc=1000, convert_rgb_to_intensity=False)

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

height = 720
width = 1280

depthscale = 1000
depth_image = np.asarray(rgbd_image.depth, dtype=np.float64) * depthscale
color_image = np.asarray(rgbd_image.color)

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

cx = 648.483276367188
cy = 359.049194335938
fx = 787.998596191406
fy = 786.333374023438

depthscale = 1000
depth_image = np.asarray(rgbd_image.depth) * 1000
U = np.asarray(list(range(width))).reshape(1, -1).repeat(720, axis=0)
V = np.asarray(list(range(height))).reshape(-1, 1).repeat(1280, axis=1)
Z = depth_image / depthscale
X = (U - cx) * Z / fx
Y = (V - cy) * Z / fy
XYZ = np.stack((X, Y, Z), axis=-1)
indices = depth_image > 0
np_points = XYZ[indices, :]
np_colors = color_image[indices, :]
color_image_projected_on_depth = np.zeros((720, 1280, 3), dtype=int)
color_image_projected_on_depth[indices, :] = color_image[indices, :]

xyz_matrix = np.zeros((720, 1280, 3), dtype=np.float64)
xyz_matrix[:] = np.nan
xyz_matrix[indices, :] = XYZ[indices, :]

real_indices = np.argwhere(indices)
l = len(real_indices)
new_img = np.zeros((720, 1280, 3), dtype=np.int)
'''
for i in range(l):
    point_u, point_v = real_indices[i]
    point_xyz = xyz_matrix[point_u, point_v]
    point_color = color_image[point_u, point_v]

    therm_focals = [1008.8578238167438196598189451667,	1020.9726220290907682227917520318]
    therm_centers = [319.690452842080,	200.489004453523]
    therm_distortion_coefs = [3.26801409578410, -296.593697284903, 0, 0, 16968.6805606598]

    rs_focals = [700.715723643176,	700.900750363357]
    rs_centers = [627.056652072525, 428.897247449207]
    rs_distortion_coefs = [0.631373966934844, -21.1797091234745, 0, 0,	146.583455319192]

    original_rs_focals = [787.998596191406, 786.333374023438]
    original_rs_centers = [648.483276367188, 359.049194335938]

    #after matlab refine


    R = np.array([
        [0.999235425597163, - 0.0120010932215866, - 0.0372093804455857],
        [0.0161048474321583, 0.993574345568653, 0.112029700155848],
        [0.0356258069500811, - 0.112543296509872, 0.993007959832068]
    ])

    t = np.array([-121.952452666493, 189.727110138437, -813.867721109518])


    def project(point_xyz, focals, centers, distortion_coefs, use_dist=False, R = np.eye, t = np.ones((3,1))):
        #distortion_coefs: k1,k2,p1,p2,k3, focals: fx,fy, centers same.
        xyz = np.array(point_xyz).reshape(-1, 1) * 1000
        t[0] = t[0] + 90   # + move left camera angle changes right
        t[1] = t[1] - 350
        #t[2] = t[2] + 150
        rotated_translated_xyz = R @ xyz + t.reshape(3,1)

        x, y = rotated_translated_xyz[0] / rotated_translated_xyz[2], rotated_translated_xyz[1] / rotated_translated_xyz[2]

        if use_dist:
            r2 = x * x + y * y;
            f = 1 + distortion_coefs[0] * r2 + distortion_coefs[1] * r2 * r2 + distortion_coefs[4] * r2 * r2 * r2;
            x, y = x * f, y * f
            dx = x + 2 * distortion_coefs[1] * x * y + distortion_coefs[3] * (r2 + 2 * x * x)
            dy = y + 2 * distortion_coefs[3] * x * y + distortion_coefs[2] * (r2 + 2 * y * y)
            x = dx;
            y = dy;

        pixel_u = x * focals[0] + centers[0] + 18.5 # + is right
        pixel_v = y * focals[1] + centers[1] + 45# + is down
        if pixel_u <= 0 or pixel_v <= 0 or pixel_u > 1279.5 or pixel_v > 719.5:
            return np.array(0),np.array(0)

        return pixel_v, pixel_u


    returned_u, returned_v = project(point_xyz, therm_focals, therm_centers, rs_distortion_coefs, False, np.eye(3), t)
    igul_u, igul_v = int(round(returned_u.item())), int(round(returned_v.item()))
    new_img[igul_u, igul_v, :] = point_color
'''
point_u, point_v = real_indices.transpose()
point_xyz = xyz_matrix[point_u, point_v].transpose()
point_color = color_image[point_u, point_v]
point_depth = depth_image[point_u, point_v]

# therm_focals = [1008.8578238167438196598189451667,	1020.9726220290907682227917520318]
therm_focals = [835.8578238167438196598189451667, 815.9726220290907682227917520318]  # hor, ver
therm_centers = [319.690452842080, 200.489004453523]
therm_distortion_coefs = [3.26801409578410, -296.593697284903, 0, 0, 16968.6805606598]

rs_focals = [700.715723643176, 700.900750363357]
rs_centers = [627.056652072525, 428.897247449207]
rs_distortion_coefs = [0.631373966934844, -21.1797091234745, 0, 0, 146.583455319192]

original_rs_focals = [787.998596191406, 786.333374023438]
original_rs_centers = [648.483276367188, 359.049194335938]

# after matlab refine


R = np.array([
    [0.999235425597163, - 0.0120010932215866, - 0.0372093804455857],
    [0.0161048474321583, 0.993574345568653, 0.112029700155848],
    [0.0356258069500811, - 0.112543296509872, 0.993007959832068]
])

t = np.array([-121.952452666493, 189.727110138437, -813.867721109518])


def project(point_xyz, focals, centers, distortion_coefs, use_dist=False, R=np.eye, t=np.ones((3, 1))):
    # distortion_coefs: k1,k2,p1,p2,k3, focals: fx,fy, centers same.
    xyz = point_xyz * 1000
    t[0] = t[0] - 50  # + move left camera angle changes right
    t[1] = t[1] - 300  # + move up camera
    # t[2] = t[2] + 150
    rotated_translated_xyz = R @ xyz + t.reshape(3, 1)

    x, y = rotated_translated_xyz[0, :] / rotated_translated_xyz[2, :], rotated_translated_xyz[1,
                                                                        :] / rotated_translated_xyz[2, :]

    if use_dist:
        r2 = x * x + y * y;
        f = 1 + distortion_coefs[0] * r2 + distortion_coefs[1] * r2 * r2 + distortion_coefs[4] * r2 * r2 * r2;
        x, y = x * f, y * f
        dx = x + 2 * distortion_coefs[1] * x * y + distortion_coefs[3] * (r2 + 2 * x * x)
        dy = y + 2 * distortion_coefs[3] * x * y + distortion_coefs[2] * (r2 + 2 * y * y)
        x = dx;
        y = dy;

    pixel_u = x * focals[0] + centers[
        0] - 37.5  # + is right (if camera moved left here it should brought left and vice versa)
    pixel_v = y * focals[1] + centers[
        1] + 18  # + is down (if camera moved down here it should brought down and vice versa)
    pixel_u[(pixel_u <= 0) | (pixel_u > 1279.5)] = 0
    pixel_v[(pixel_v <= 0) | (pixel_v > 719.5)] = 0

    return pixel_v, pixel_u


returned_u, returned_v = project(point_xyz, therm_focals, therm_centers, rs_distortion_coefs, False, np.eye(3), t)
igul_u, igul_v = np.round(returned_u).astype(int), np.round(returned_v).astype(int)
new_img[igul_u, igul_v, :] = point_color

projected_depth = np.zeros_like(depth_image)
projected_depth[igul_u, igul_v] = point_depth

points_2d = np.load("points_2d.npz")["arr_0"]
camera_indices = np.load("camera_indices.npz")["arr_0"]
point_indices = np.load("point_indices.npz")["arr_0"]
points_3d_left = np.load("points_3d_left.npz")["arr_0"]



color_raw = o3d.geometry.Image(np_left_im)
depth_raw = o3d.geometry.Image(projected_depth[:np_left_im.shape[0], :np_left_im.shape[1]])

rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw, depth_raw, depth_scale=1000.0, depth_trunc=1000, convert_rgb_to_intensity=False)

height = 512
width = 640

depthscale = 1000
depth_image = np.asarray(rgbd_image.depth, dtype=np.float64) * depthscale
color_image = np.asarray(rgbd_image.color)

cx=319.690452842080
cy=200.489004453523
fx=835.8578238167438196598189451667
fy=815.9726220290907682227917520318

depthscale = 1000
depth_image = np.asarray(rgbd_image.depth) * 1000
U = np.asarray(list(range(width))).reshape(1, -1).repeat(height, axis=0)
V = np.asarray(list(range(height))).reshape(-1, 1).repeat(width, axis=1)
Z = depth_image / depthscale
X = (U - cx) * Z / fx
Y = (V - cy) * Z / fy
XYZ = np.stack((X, Y, Z), axis=-1)

points_3d_right = XYZ[points_2d[points_2d.shape[0] // 2:, 0],
                     points_2d[points_2d.shape[0] // 2:, 1], :]


points_3d = np.concatenate((points_3d_left, points_3d_right))
np.savez("points_3d.npz", points_3d)
os.remove("points_3d_left.npz")
