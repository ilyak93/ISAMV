# two approaches: A. to project depth directly onto  left and right B. to deproject and project onto left and right
# also with tangential distortion cooefs and without

# first approach:
import cv2
import open3d as o3d
import numpy as np
from matplotlib.pyplot import imshow, show

#454 - 700
#np_color_im = cv2.imread("G:/Vista_project/finish_deep/aligned_rs/650color.png", cv2.IMREAD_COLOR)
#np_color_im = cv2.cvtColor(np_color_im, cv2.COLOR_BGR2RGB)
#np_depth_im = cv2.imread("G:/Vista_project/finish_deep/aligned_rs/650depth.png", cv2.IMREAD_ANYDEPTH)
#np_left_im = cv2.imread("G:/Vista_project/fin ish_deep/left_resized/650color.png", cv2.IMREAD_ANYDEPTH)

np_color_im = cv2.imread("G:/Vista_project/4/aligned_rs/41color.png", cv2.IMREAD_COLOR)
np_color_im = cv2.cvtColor(np_color_im, cv2.COLOR_BGR2RGB)
np_depth_im = cv2.imread("G:/Vista_project/4/aligned_rs/41depth.png", cv2.IMREAD_ANYDEPTH)
np_left_im = cv2.imread("G:/Vista_project/4/right_resized/41color.png", cv2.IMREAD_ANYDEPTH)

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

new_points = []
new_colors = []
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
        t[0] = t[0] - 80   # + move left camera angle changes right
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

        pixel_u = x * focals[0] + centers[0] - 8.65 # + is right
        pixel_v = y * focals[1] + centers[1] + 30.5# + is down
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
    xyz = point_xyz * 1000
    t[0] = t[0] - 80   # + move left camera angle changes right
    t[1] = t[1] - 350
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

    pixel_u = x * focals[0] + centers[0] - 8.65 # + is right
    pixel_v = y * focals[1] + centers[1] + 30.5# + is down
    pixel_u[(pixel_u <= 0) | (pixel_u > 1279.5)] = 0
    pixel_v[(pixel_v <= 0) | (pixel_v > 719.5)] = 0

    return pixel_v, pixel_u


returned_u, returned_v = project(point_xyz, therm_focals, therm_centers, rs_distortion_coefs, False, np.eye(3), t)
igul_u, igul_v = np.round(returned_u).astype(int), np.round(returned_v).astype(int)
new_img[igul_u, igul_v, :] = point_color
imshow(np_color_im)
show()
imshow(new_img)
show()
imshow(np_left_im)
show()
imshow(color_image_projected_on_depth)
show()
print()






