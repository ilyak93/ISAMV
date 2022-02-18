import copy

import cv2
import matplotlib
import numpy as np
import scipy
from open3d.cpu.pybind.geometry import KDTreeSearchParam, KDTreeSearchParamKNN

from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering, OPTICS, MeanShift, SpectralClustering
from matplotlib import pyplot as plt

# #############################################################################


import numpy as np
from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt
import cv2

colors_dict = {}
for name, hex in matplotlib.colors.cnames.items():
    colors_dict[name] = matplotlib.colors.to_rgb(hex)
colors_dict.pop("black")
num_of_colors = len(colors_dict);
my_colors_dict = dict()

for key, color in enumerate(colors_dict.values()):
    my_colors_dict[key] = [t for t in color]

#import one image and create pcd
'''
color_raw1 = o3d.io.read_image("ex1.png")
depth_raw1 = o3d.io.read_image("ex1_d.png")


rgbd_image1 = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw1, depth_raw1, depth_scale=1000.0, depth_trunc=5, convert_rgb_to_intensity=False)
print(rgbd_image1)

plt.subplot(1, 2, 1)
plt.title('Redwood grayscale image')
plt.imshow(rgbd_image1.color)
plt.subplot(1, 2, 2)
plt.title('Redwood depth image')
plt.imshow(rgbd_image1.depth)
plt.show()

#intrinsic = read_pinhole_camera_intrinsic("real_sense_intrinsic")
intrinsic = o3d.camera.PinholeCameraIntrinsic(width=1280, height=720,
                                              fx=787.998596191406,
                                              fy=786.333374023438,
                                              cx=648.483276367188,
                                              cy=359.049194335938)

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image1, intrinsic)
# Flip it, otherwise the pointcloud will be upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
print("original ds pointcloud")
pcd = pcd.voxel_down_sample(voxel_size=0.03)
o3d.visualization.draw_geometries([pcd])
'''


'''
#XYZ clustering
points = np.asarray(pcd.points)

db = DBSCAN(eps=0.05, min_samples=2, metric='euclidean', algorithm='auto') #eps 5 is too much
db.fit(points)
labels = db.labels_

colors = []
for i in range(labels.shape[0]):
    if labels[i] == -1:
        colors.append([0,0,0])
    else:
        color_idx = labels[i] % num_of_colors
        colors.append(my_colors_dict[color_idx])

labels_colors = np.asarray(colors)

print("unique my dbscan number of clusters:" + str(np.unique(labels).shape))

clustered_pcd = o3d.geometry.PointCloud()
clustered_pcd.points = o3d.utility.Vector3dVector(points)
clustered_pcd.colors = o3d.utility.Vector3dVector(labels_colors)
print("downsampled 0.01 voxel and sklearn sbcanned clustered pointcloud")
o3d.visualization.draw_geometries([clustered_pcd])
'''

'''
#colors of XYZ clustering
points = np.asarray(pcd.points)
pt_colors = np.asarray(pcd.colors)


db = DBSCAN(eps=0.005, min_samples=5, metric='euclidean', algorithm='auto') #eps 5 is too much
db.fit(pt_colors)
labels = db.labels_

colors_dict = {}
for name, hex in matplotlib.colors.cnames.items():
    colors_dict[name] = matplotlib.colors.to_rgb(hex)
colors_dict.pop("black")
num_of_colors = len(colors_dict);
my_colors_dict = dict()

for key, color in enumerate(colors_dict.values()):
    my_colors_dict[key] = [t for t in color]

colors = []
for i in range(labels.shape[0]):
    if labels[i] == -1:
        colors.append([0,0,0])
    else:
        color_idx = labels[i] % num_of_colors
        colors.append(my_colors_dict[color_idx])

labels_colors = np.asarray(colors)

print("unique my dbscan number of clusters:" + str(np.unique(labels).shape))

clustered_pcd = o3d.geometry.PointCloud()
clustered_pcd.points = o3d.utility.Vector3dVector(points)
clustered_pcd.colors = o3d.utility.Vector3dVector(labels_colors)
print("downsampled 0.01 voxel and sklearn sbcanned clustered pointcloud")
o3d.visualization.draw_geometries([clustered_pcd])
'''

'''
#Normals clustering
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

points = np.asarray(pcd.points)
normals = np.asarray(pcd.colors)

db = DBSCAN(eps=0.005, min_samples=2, metric='euclidean', algorithm='auto') #eps 5 is too much
db.fit(normals)
labels = db.labels_

colors_dict = {}
for name, hex in matplotlib.colors.cnames.items():
    colors_dict[name] = matplotlib.colors.to_rgb(hex)
colors_dict.pop("black")
num_of_colors = len(colors_dict);
my_colors_dict = dict()

for key, color in enumerate(colors_dict.values()):
    my_colors_dict[key] = [t for t in color]

colors = []
for i in range(labels.shape[0]):
    if labels[i] == -1:
        colors.append([0,0,0])
    else:
        color_idx = labels[i] % num_of_colors
        colors.append(my_colors_dict[color_idx])

labels_colors = np.asarray(colors)

print("unique my dbscan number of clusters:" + str(np.unique(labels).shape))

clustered_pcd = o3d.geometry.PointCloud()
clustered_pcd.points = o3d.utility.Vector3dVector(points)
clustered_pcd.colors = o3d.utility.Vector3dVector(labels_colors)
print("downsampled 0.01 voxel and sklearn sbcanned clustered pointcloud")
o3d.visualization.draw_geometries([clustered_pcd])
'''


#Clustering velocities
#import 2 images, create pcds and vizualize,
# compute velocities and cluster
'''
intrinsic = o3d.camera.PinholeCameraIntrinsic(width=1280, height=720,
                                              fx=787.998596191406,
                                              fy=786.333374023438,
                                              cx=648.483276367188,
                                              cy=359.049194335938)

color_raw1 = o3d.io.read_image("ex1.png")
depth_raw1 = o3d.io.read_image("ex1_d.png")


rgbd_image1 = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw1, depth_raw1, depth_scale=1000.0, depth_trunc=5, convert_rgb_to_intensity=False)
print(rgbd_image1)

pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image1, intrinsic)
pcd1.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
pcd1 = pcd1.voxel_down_sample(voxel_size=0.02)
o3d.visualization.draw_geometries([pcd1])

color_raw2 = o3d.io.read_image("ex2.png")
depth_raw2 = o3d.io.read_image("ex2_d.png")


rgbd_image2 = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw2, depth_raw2, depth_scale=1000.0, depth_trunc=5, convert_rgb_to_intensity=False)
print(rgbd_image2)

pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image2, intrinsic)
pcd2.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
pcd2 = pcd2.voxel_down_sample(voxel_size=0.02)
o3d.visualization.draw_geometries([pcd2])

source_temp = copy.deepcopy(pcd1)
target_temp = copy.deepcopy(pcd2)
source_temp.paint_uniform_color([1, 0.706, 0])
target_temp.paint_uniform_color([0, 0.651, 0.929])
o3d.visualization.draw_geometries([source_temp, target_temp])

def my_compute_point_cloud_distance(points, target):
    kdTree = o3d.geometry.KDTreeFlann()
    kdTree.set_geometry(target)
    distances = np.zeros(points.shape[0])
    indices = np.zeros(points.shape[0])
    for i, point in enumerate(points):
        ind_dist = kdTree.search_knn_vector_3d(point, 1)
        if ind_dist[0] == 0:
            distances[i] = 0
            indices[i] = -1
        else:
            distances[i] = np.sqrt(np.asarray(ind_dist[2])[0])
            indices[i] = np.asarray(ind_dist[1])[0]
    return distances, indices

dd, ii = my_compute_point_cloud_distance(np.asarray(pcd1.points), pcd2)

velocities = np.asarray(pcd2.points)[ii.astype(int),:] - np.asarray(pcd1.points)[:,:]

pcd_velocities_viz = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image1, intrinsic)
# Flip it, otherwise the pointcloud will be upside down
pcd_velocities_viz.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
print("original pointcloud")
pcd_velocities_viz = pcd_velocities_viz.voxel_down_sample(voxel_size=0.02)
pcd_velocities_viz.normals = o3d.utility.Vector3dVector(velocities)
#o3d.visualization.draw_geometries([pcd1])
o3d.visualization.draw_geometries([pcd_velocities_viz], point_show_normal=True)

db = DBSCAN(eps=0.0091, min_samples=2, metric='euclidean', algorithm='auto') #eps 5 is too much
db.fit(velocities)
labels = db.labels_
colors = []
for i in range(labels.shape[0]):
    if labels[i] == -1:
        colors.append([0,0,0])
    else:
        color_idx = labels[i] % num_of_colors
        colors.append(my_colors_dict[color_idx])

labels_colors = np.asarray(colors)
#print(np.unique(labels))
print("unique my dbscan number of clusters:" + str(np.unique(labels).shape))
#hist = np.histogram(labels)
#plt.hist(labels, bins='auto')
#plt.show()

points = np.asarray(pcd1.points)

clustered_pcd1 = o3d.geometry.PointCloud()
clustered_pcd1.points = o3d.utility.Vector3dVector(points)
clustered_pcd1.colors = o3d.utility.Vector3dVector(labels_colors)
print("downsampled 0.01 voxel and sklearn sbcanned clustered pointcloud")
o3d.visualization.draw_geometries([clustered_pcd1])
'''

#distances clustering
'''
intrinsic = o3d.camera.PinholeCameraIntrinsic(width=1280, height=720,
                                              fx=787.998596191406,
                                              fy=786.333374023438,
                                              cx=648.483276367188,
                                              cy=359.049194335938)

color_raw1 = o3d.io.read_image("ex1.png")
depth_raw1 = o3d.io.read_image("ex1_d.png")


rgbd_image1 = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw1, depth_raw1, depth_scale=1000.0, depth_trunc=5, convert_rgb_to_intensity=False)
print(rgbd_image1)

pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image1, intrinsic)
pcd1.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
pcd1 = pcd1.voxel_down_sample(voxel_size=0.02)
o3d.visualization.draw_geometries([pcd1])

color_raw2 = o3d.io.read_image("ex2.png")
depth_raw2 = o3d.io.read_image("ex2_d.png")


rgbd_image2 = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw2, depth_raw2, depth_scale=1000.0, depth_trunc=5, convert_rgb_to_intensity=False)
print(rgbd_image2)

pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image2, intrinsic)
pcd2.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
pcd2 = pcd2.voxel_down_sample(voxel_size=0.02)
o3d.visualization.draw_geometries([pcd2])

source_temp = copy.deepcopy(pcd1)
target_temp = copy.deepcopy(pcd2)
source_temp.paint_uniform_color([1, 0.706, 0])
target_temp.paint_uniform_color([0, 0.651, 0.929])
o3d.visualization.draw_geometries([source_temp, target_temp])

def my_compute_point_cloud_distance(points, target):
    kdTree = o3d.geometry.KDTreeFlann()
    kdTree.set_geometry(target)
    distances = np.zeros(points.shape[0])
    indices = np.zeros(points.shape[0])
    for i, point in enumerate(points):
        ind_dist = kdTree.search_knn_vector_3d(point, 1)
        if ind_dist[0] == 0:
            distances[i] = 0
            indices[i] = -1
        else:
            distances[i] = np.sqrt(np.asarray(ind_dist[2])[0])
            indices[i] = np.asarray(ind_dist[1])[0]
    return distances, indices

dd, ii = my_compute_point_cloud_distance(np.asarray(pcd1.points), pcd2)

distances = np.sum(np.asarray(pcd2.points)[ii.astype(int),:] - np.asarray(pcd1.points)[:,:], axis=1).reshape(-1,1)


db = DBSCAN(eps=0.0005, min_samples=5, metric='euclidean', algorithm='auto') #eps 5 is too much
db.fit(distances)
labels = db.labels_
colors = []
for i in range(labels.shape[0]):
    if labels[i] == -1:
        colors.append([0,0,0])
    else:
        color_idx = labels[i] % num_of_colors
        colors.append(my_colors_dict[color_idx])

labels_colors = np.asarray(colors)
#print(np.unique(labels))
print("unique my dbscan number of clusters:" + str(np.unique(labels).shape))
#hist = np.histogram(labels)
#plt.hist(labels, bins='auto')
#plt.show()

points = np.asarray(pcd1.points)

clustered_pcd1 = o3d.geometry.PointCloud()
clustered_pcd1.points = o3d.utility.Vector3dVector(points)
clustered_pcd1.colors = o3d.utility.Vector3dVector(labels_colors)
print("downsampled 0.01 voxel and sklearn sbcanned clustered pointcloud")
o3d.visualization.draw_geometries([clustered_pcd1])
'''