import time
import urllib
import bz2
import os
from ctypes import windll, wintypes

import open3d as o3d

import cv2
import numpy as np
from scipy.optimize import least_squares

BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/ladybug/"
FILE_NAME = "problem-49-7776-pre.txt.bz2"
URL = BASE_URL + FILE_NAME

if not os.path.isfile(FILE_NAME):
    urllib.request.urlretrieve(URL, FILE_NAME)

def read_bal_data(file_name):
    with bz2.open(file_name, "rt") as file:
        n_cameras, n_points, n_observations = map(
            int, file.readline().split())

        camera_indices = np.empty(n_observations, dtype=int)
        point_indices = np.empty(n_observations, dtype=int)
        points_2d = np.empty((n_observations, 2))

        for i in range(n_observations):
            camera_index, point_index, x, y = file.readline().split()
            camera_indices[i] = int(camera_index)
            point_indices[i] = int(point_index)
            points_2d[i] = [float(x), float(y)]

        camera_params = np.empty(n_cameras * 9)
        for i in range(n_cameras * 9):
            camera_params[i] = float(file.readline())
        camera_params = camera_params.reshape((n_cameras, -1))

        points_3d = np.empty(n_points * 3)
        for i in range(n_points * 3):
            points_3d[i] = float(file.readline())
        points_3d = points_3d.reshape((n_points, -1))

    return camera_params, points_3d, camera_indices, point_indices, points_2d

camera_params, points_3d, camera_indices, point_indices, points_2d = read_bal_data(FILE_NAME)


def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.

    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

n_cameras = camera_params.shape[0]
n_points = points_3d.shape[0]

n = 9 * n_cameras + 3 * n_points
m = 2 * points_2d.shape[0]

print("n_cameras: {}".format(n_cameras))
print("n_points: {}".format(n_points))
print("Total number of parameters: {}".format(n))
print("Total number of residuals: {}".format(m))

def project_a(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    n = np.sum(points_proj**2, axis=1)
    r = 1 + k1 * n + k2 * n**2
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj




def my_project_a(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    #points_proj = rotate(points, camera_params[:, :3])
    points_proj = points
    points_proj += camera_params[:, 4:7]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    fx = camera_params[:, 0]
    fy = camera_params[:, 1]
    cpx = camera_params[:, 2]
    cpy = camera_params[:, 3]
    points_proj[:, 0] *= fx
    points_proj[:, 0] += cpx
    points_proj[:, 1] *= fy
    points_proj[:, 0] += cpy
    return points_proj


def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """Compute residuals.

    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params[n_cameras * 9:].reshape((n_points, 3))
    points_proj = my_project_a(points_3d[point_indices], camera_params[camera_indices])
    return (points_proj - points_2d).ravel()

from scipy.sparse import lil_matrix

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(9):
        A[2 * i, camera_indices * 9 + s] = 1
        A[2 * i + 1, camera_indices * 9 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1

    return A


import matplotlib.pyplot as plt
'''
x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)
plt.plot(f0)

A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)

t0 = time.time()
res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                    args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
t1 = time.time()

print("Optimization took {0:.0f} seconds".format(t1 - t0))

plt.plot(res.fun)
plt.show()
'''

points_2d = np.load("points_2d.npz")["arr_0"]
camera_indices = np.load("camera_indices.npz")["arr_0"]
point_indices = np.load("point_indices.npz")["arr_0"]
points_3d = np.load("points_3d.npz")["arr_0"]
camera_params = camera_params[0:2, :]

camera_params[:, 0] = 2100
camera_params[:, 1] = 2100
camera_params[:, 2] = 320
camera_params[:, 3] = 200
camera_params[:, 4] = 0
camera_params[:, 5] = 0
camera_params[:, 6] = -800

n_cameras = camera_params.shape[0]
n_points = points_3d.shape[0]

n = 9 * n_cameras + 3 * n_points
m = 2 * points_2d.shape[0]

print("n_cameras: {}".format(n_cameras))
print("n_points: {}".format(n_points))
print("Total number of parameters: {}".format(n))
print("Total number of residuals: {}".format(m))

x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)
plt.plot(f0)
plt.show()

A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)

t0 = time.time()
res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                    args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
t1 = time.time()

print("Optimization took {0:.0f} seconds".format(t1 - t0))

plt.plot(res.fun)

plt.show()

def prettylist(l):
    return '[%s]' % ', '.join("%4.1e" % f for f in l)

print('Before:')
print('cam0: {}'.format(prettylist(x0[0:9])))
print('cam1: {}'.format(prettylist(x0[9:18])))

print('After:')
print('cam0: {}'.format(prettylist(res.x[0:9])))
print('cam1: {}'.format(prettylist(res.x[9:18])))

print('Before:')
print('3D 1: {}'.format(prettylist(x0[18:21])))
print('3D 2: {}'.format(prettylist(x0[21:24])))

print('After:')
print('3D 1: {}'.format(prettylist(res.x[18:21])))
print('3D 2: {}'.format(prettylist(res.x[21:24])))




#Projection check

from functools import cmp_to_key
def winsort(data):
    _StrCmpLogicalW = windll.Shlwapi.StrCmpLogicalW
    _StrCmpLogicalW.argtypes = [wintypes.LPWSTR, wintypes.LPWSTR]
    _StrCmpLogicalW.restype  = wintypes.INT

    cmp_fnc = lambda psz1, psz2: _StrCmpLogicalW(psz1, psz2)
    return sorted(data, key=cmp_to_key(cmp_fnc))

#454 - 700
#np_color_im = cv2.imread("G:/Vista_project/finish_deep/aligned_rs/650color.png", cv2.IMREAD_COLOR)
#np_color_im = cv2.cvtColor(np_color_im, cv2.COLOR_BGR2RGB)
#np_depth_im = cv2.imread("G:/Vista_project/finish_deep/aligned_rs/650depth.png", cv2.IMREAD_ANYDEPTH)
#np_left_im = cv2.imread("G:/Vista_project/finish_deep/left_resized/650color.png", cv2.IMREAD_ANYDEPTH)

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
file_num = 0
files = 4
#file_num = file_num - 1
thermal_file_path = gen_path + str(dataset_num) + "_ready/" + \
                    quartet_folder_name + "/" + \
                    all_files[file_num * files + 2]
if "left" in thermal_file_path:
    left_file_path = thermal_file_path
    right_file_path = gen_path + str(dataset_num) + "_ready/" + \
                      quartet_folder_name + "/" + all_files[file_num * files + 3]
else:
    right_file_path = thermal_file_path
    left_file_path = gen_path + str(dataset_num) + "_ready/" + \
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
np_right_im = cv2.imread(right_file_path, cv2.IMREAD_ANYDEPTH)


color_raw = o3d.geometry.Image(np_color_im)
depth_raw = o3d.geometry.Image(np_depth_im)

rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw, depth_raw, depth_scale=1000.0, depth_trunc=1000, convert_rgb_to_intensity=False)


height = 720
width = 1280

depthscale = 1000
depth_image = np.asarray(rgbd_image.depth, dtype=np.float64) * depthscale
color_image = np.asarray(rgbd_image.color)


cx=648.483276367188
cy=359.049194335938
fx=787.998596191406
fy=786.333374023438

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
color_image_projected_on_depth = np.zeros((720,1280,3), dtype=int)
color_image_projected_on_depth[indices, :] = color_image[indices, :]

xyz_matrix = np.zeros((720, 1280, 3), dtype=np.float64)
xyz_matrix[:] = np.nan
xyz_matrix[indices, :] = XYZ[indices, :]

real_indices = np.argwhere(indices)
l = len(real_indices)
new_img = np.zeros((720, 1280, 3), dtype=np.int)
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


def project(point_xyz, focals, centers, distortion_coefs, use_dist=False, R = np.eye, t = np.ones((3,1))):
    #distortion_coefs: k1,k2,p1,p2,k3, focals: fx,fy, centers same.
    xyz = point_xyz * 1000
    t[0] = t[0] - 50# + move left camera angle changes left
    t[1] = t[1] - 350 # + move up camera
    #t[2] = t[2] + 150
    rotated_translated_xyz = R @ xyz + t.reshape(3,1)

    x, y = rotated_translated_xyz[0, :] / rotated_translated_xyz[2, :], rotated_translated_xyz[1, :] / rotated_translated_xyz[2, :]

    if use_dist:
        r2 = x * x + y * y;
        f = 1 + distortion_coefs[0] * r2 + distortion_coefs[1] * r2 * r2 + distortion_coefs[4] * r2 * r2 * r2;
        x, y = x * f, y * f
        dx = x + 2 * distortion_coefs[1] * x * y + distortion_coefs[3] * (r2 + 2 * x * x)
        dy = y + 2 * distortion_coefs[3] * x * y + distortion_coefs[2] * (r2 + 2 * y * y)
        x = dx;
        y = dy;

    pixel_u = x * focals[0] + centers[0] + 12.5 # + is right (if camera moved left here it should brought left and vice versa)
    pixel_v = y * focals[1] + centers[1] + 24  # + is down (if camera moved down here it should brought down and vice versa)
    pixel_u[(pixel_u <= 0) | (pixel_u > 1279.5)] = 0
    pixel_v[(pixel_v <= 0) | (pixel_v > 719.5)] = 0

    return pixel_v, pixel_u


returned_u, returned_v = project(point_xyz, therm_focals, therm_centers, rs_distortion_coefs, False, np.eye(3), t)
igul_u, igul_v = np.round(returned_u).astype(int), np.round(returned_v).astype(int)
new_img[igul_u, igul_v, :] = point_color

projected_depth = np.zeros_like(depth_image)
projected_depth[igul_u, igul_v] = point_depth

def median_filter_handmode_1D(image, ksize):
    # Create a Gaussian kernel

    # Apply padding to the image
    pad = ksize // 2
    padded_image = np.zeros((image.shape[0] + pad * 2,
                             image.shape[1] + pad * 2), dtype=np.float32)

    padded_image[:, :] = np.pad(image[:, :], pad, mode='constant')

    # Convolve the padded image with the Gaussian kernel
    filtered_image = padded_image
    height, width = image.shape

    for i in range(height - ksize + 1):
        for j in range(width - ksize + 1):
            if padded_image[i, j] == 0:
                patch = image[i:i + ksize, j:j + ksize]
                med = np.quantile(patch, 0.75)
                filtered_image[i, j] = med


    return filtered_image



filtered_projected_depth = median_filter_handmode_1D(
    projected_depth[:np_left_im.shape[0],
    :np_left_im.shape[1]], 3)[1:-1, 1:-1]



height = 512
width = 640

cx=319.690452842080
cy=200.489004453523
fx=835.8578238167438196598189451667
fy=815.9726220290907682227917520318

depthscale = 1000
depth_image = filtered_projected_depth
U = np.asarray(list(range(width))).reshape(1, -1).repeat(height, axis=0)
V = np.asarray(list(range(height))).reshape(-1, 1).repeat(width, axis=1)
Z = depth_image
X = (U - cx) * Z / fx
Y = (V - cy) * Z / fy
XYZ = np.stack((X, Y, Z), axis=-1)

projected_to_thermal2 = my_project_a(XYZ.reshape(-1, 3),
                                  np.repeat(res.x[0:9].reshape(1, -1),
                                            640*512, axis=0)).reshape((512,640,2))

projected_to_thermal2[:,:, 0] += max(-projected_to_thermal2[:,:, 0].min(), 0)
projected_to_thermal2[:,:, 1] += max(-projected_to_thermal2[:,:, 1].min(), 0)
projected_to_thermal2 = projected_to_thermal2.astype(int)

right_depth = np.zeros((1000, 1000))

right_depth[projected_to_thermal2[:, :, 1],
            projected_to_thermal2[:, :, 0]] = XYZ[:, :, 2]

plt.imshow(right_depth)
plt.show()





