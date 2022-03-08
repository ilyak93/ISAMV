# skeleton source https://github.com/opencv/opencv/blob/master/samples/python/stereo_match.py
import numpy as np
import cv2
import scipy
from matplotlib import pyplot as plt
import open3d as o3d
from scipy import io


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

print('loading images...')
imgL_depth = cv2.imread('G:/Vista_project/new_motion/left/250_761653140100.png', cv2.IMREAD_ANYDEPTH)
imgR_depth = cv2.imread('G:/Vista_project/new_motion/right/250_761653091000.png', cv2.IMREAD_ANYDEPTH)
imgL = np.round((imgL_depth - imgL_depth.min()) / (imgL_depth.max() - imgL_depth.min()) * 256).astype(np.uint8);
imgR = np.round((imgR_depth - imgR_depth.min()) / (imgR_depth.max() - imgR_depth.min()) * 256).astype(np.uint8);

# disparity range tuning
# https://docs.opencv.org/trunk/d2/d85/classcv_1_1StereoSGBM.html
window_size = 10
min_disp = 1
num_disp = 128 - min_disp
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

print('computing disparity...')
# disparity = stereo.compute(imgL, imgR)
#imgL[imgL_depth < 10000] = 0
#imgR[imgR_depth < 10000] = 255

plt.imshow(imgL, 'gray')
plt.show()

plt.imshow(imgR, 'gray')
plt.show()

disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
plt.imshow(disparity, 'gray')
plt.show()

disparity[disparity>0] = 128 - disparity[disparity>0]


io.savemat('./dis.mat', {'disparity': disparity})

# plt.imshow(imgL, 'gray')
plt.imshow(disparity, 'gray')
#plt.colorbar()
# plt.imshow('disparity', (disparity - min_disp) / num_disp)
plt.show()


print("\nGenerating the 3D map ...")
h, w = imgL.shape[:2]
focal_length = 0.8*w

# Perspective transformation matrix
Q = np.float32([[1, 0, 0, -w/2.0],
                [0, -1, 0,  h/2.0],
                [0, 0, 0, -focal_length],
                [0, 0, 1, 0]])
'''
Q = np.float32([[1,	0,	0,	0],
               [0,	1,	0,	0],
               [0,	0,	1,	-0.0066789887],
               [22.707514,	-207.43671,	1450.2950,	1]])

#Q = np.eye(4)
'''
points_3D = cv2.reprojectImageTo3D(disparity, Q)
points_3D[:,:,1] = points_3D[:,:,1] / 600 + 0.0025
points_3D[:,:,0] = points_3D[:,:,0] / 600 - 0.001
points_3D[:,:,2] = points_3D[:,:,2] / 50 + 0.07
filter_mask = points_3D[:,:,2] > -5 / 50 + 0.07
points_3D[:,:,2][points_3D[:,:,2] < -5  / 50 + 0.07] = -np.inf

#points_3D = scipy.io.loadmat('points.mat')['points3D']

#points_3D[:,:,1] = points_3D[:,:,1] / -1

colors = cv2.cvtColor(imgL,cv2.COLOR_GRAY2RGB)
mask_map = disparity > 0
mask_map = mask_map & filter_mask #& (imgL_depth > 2500)
output_points = points_3D[mask_map]
output_colors = colors[mask_map]

plt.imshow(points_3D[:,:,2])
plt.colorbar()
plt.show()

'''
points = output_points.reshape(-1, 3)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
o3d.visualization.draw_geometries([pcd])
'''

print("\nCreating the output file ...\n")
output_file = "3dmap.ply"
create_output(output_points, output_colors, output_file)

print("Load a ply point cloud, print it, and render it")
pcd_therm = o3d.io.read_point_cloud("./3dmap.ply")
o3d.visualization.draw_geometries([pcd_therm])

#cv2.imshow('Left Image', image_left)
#cv2.imshow('Right Image', image_right)
#cv2.imshow('Disparity Map', (disparity_map - min_disp) / num_disp)
#cv2.waitKey()
#cv2.destroyAllWindows()

import numpy as np
from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt
import cv2
from open3d.cpu.pybind.io import read_pinhole_camera_intrinsic

color_frame = 'G:/Vista_project/new_motion/sync/250color.png'
color_im = Image.fromarray(np.array(Image.open(color_frame)).astype("uint8"))
print("Image mode: ", color_im.mode)
np_color_im = np.array(Image.open(color_frame)).astype("uint8")

depth_frame = 'G:/Vista_project/new_motion/sync/250depth.png'
depth_im = Image.fromarray(np.array(Image.open(depth_frame)).astype("uint16"))
print("Image mode: ", depth_im.mode)
np_depth_im = np.array(Image.open(depth_frame)).astype("uint16")



color_raw = o3d.geometry.Image(np_color_im)
depth_raw = o3d.geometry.Image(np_depth_im)

'''
depth_raw = o3d.geometry.Image(np.array(cv2.imread("0depth_aligned.png", cv2.IMREAD_ANYDEPTH)))
plt.imshow(depth_raw)
plt.show()
dd = np.asarray(depth_raw)
'''

rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw, depth_raw, depth_scale=1000.0, depth_trunc=5, convert_rgb_to_intensity=False)
print(rgbd_image)

plt.subplot(1, 2, 1)
plt.title('Redwood grayscale image')
plt.imshow(rgbd_image.color)
plt.subplot(1, 2, 2)
plt.title('Redwood depth image')
plt.imshow(rgbd_image.depth)
plt.show()

#intrinsic = read_pinhole_camera_intrinsic("real_sense_intrinsic")
intrinsic = o3d.camera.PinholeCameraIntrinsic(width=1280, height=720,
                                              fx=787.998596191406,
                                              fy=786.333374023438,
                                              cx=648.483276367188,
                                              cy=359.049194335938)

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
o3d.visualization.draw_geometries([pcd])
o3d.visualization.draw_geometries([pcd, pcd_therm])

