import numpy as np
from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt
import cv2
from open3d.cpu.pybind.io import read_pinhole_camera_intrinsic

color_frame = '0color_aligned.png'
color_im = Image.fromarray(np.array(Image.open(color_frame)).astype("uint8"))
print("Image mode: ", color_im.mode)

depth_frame = '0depth_aligned.png'
depth_im = Image.fromarray(np.array(Image.open(depth_frame)).astype("uint16"))
print("Image mode: ", depth_im.mode)



color_raw = o3d.io.read_image("0color_aligned.png")
depth_raw = o3d.io.read_image("0depth_aligned.png")

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

'''
raw_depth = o3d.io.read_image('rs_3d_d.png')
pcd = o3d.geometry.PointCloud.create_from_depth_image(raw_depth,intrinsic)
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
print(np.asarray(pcd.points)[1,:])
o3d.visualization.draw_geometries([pcd])
'''