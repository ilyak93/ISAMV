import copy

import cv2
import numpy as np
import scipy
import matplotlib
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering, OPTICS, MeanShift, SpectralClustering


from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt



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

pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image1, intrinsic)
# Flip it, otherwise the pointcloud will be upside down
pcd1.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
print("original pointcloud")
pcd1 = pcd1.voxel_down_sample(voxel_size=0.02)
pcd1.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
o3d.visualization.draw_geometries([pcd1])
o3d.visualization.draw_geometries([pcd1], point_show_normal=True)

points_weight = 10
color_weight = 50 # 50
normals_weight = 5 # 5
points = np.asarray(pcd1.points) * points_weight
colors = np.asarray(pcd1.colors) * color_weight
normals = np.asarray(pcd1.normals) * normals_weight

z_min = points[:,2].min()
z_max = points[:,2].max()
x_min = points[:,0].min()
x_max = points[:,0].max()
y_min = points[:,1].min()
y_max = points[:,1].max()

c1_min = colors[:,2].min()
c1_max = colors[:,2].max()
c2_min = colors[:,0].min()
c2_max = colors[:,0].max()
c3_min = colors[:,1].min()
c3_max = colors[:,1].max()

n1_min = normals[:,2].min()
n1_max = normals[:,2].max()
n2_min = normals[:,0].min()
n2_max = normals[:,0].max()
n3_min = normals[:,1].min()
n3_max = normals[:,1].max()


features = np.concatenate((points, colors, normals), axis=1)
#features = np.concatenate((points, colors), axis=1)

colors_dict = {}
for name, hex in matplotlib.colors.cnames.items():
    colors_dict[name] = matplotlib.colors.to_rgb(hex)
colors_dict.pop("black")
num_of_colors = len(colors_dict);
my_colors_dict = dict()

for key, color in enumerate(colors_dict.values()):
    my_colors_dict[key] = [t for t in color]


db = DBSCAN(eps=2, min_samples=2, metric='euclidean', algorithm='auto') #eps 5 is too much
db.fit(features)
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

clustered_pcd1 = o3d.geometry.PointCloud()
clustered_pcd1.points = o3d.utility.Vector3dVector(points)
clustered_pcd1.colors = o3d.utility.Vector3dVector(labels_colors)
print("downsampled 0.01 voxel and sklearn sbcanned clustered pointcloud")
o3d.visualization.draw_geometries([clustered_pcd1])



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

dists = pcd1.compute_point_cloud_distance(pcd2)

dists = np.asarray(dists)
ind = np.where(dists > 0.1)[0]
pcd1_without_pcd2 = pcd1.select_by_index(ind)
o3d.visualization.draw_geometries([pcd1_without_pcd2])

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

vz_min = velocities[:,2].min()
vz_max = velocities[:,2].max()
vx_min = velocities[:,0].min()
vx_max = velocities[:,0].max()
vy_min = velocities[:,1].min()
vy_max = velocities[:,1].max()

pcd_velocities_viz = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image1, intrinsic)
# Flip it, otherwise the pointcloud will be upside down
pcd_velocities_viz.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
print("original pointcloud")
pcd_velocities_viz = pcd_velocities_viz.voxel_down_sample(voxel_size=0.02)
pcd_velocities_viz.normals = o3d.utility.Vector3dVector(velocities)
#o3d.visualization.draw_geometries([pcd1])
o3d.visualization.draw_geometries([pcd_velocities_viz], point_show_normal=True)

points_weight = 10
color_weight = 50 # 50
normals_weight = 5 # 5
velocities_weight = 50

points = np.asarray(pcd1.points) * points_weight
colors = np.asarray(pcd1.colors) * color_weight
normals = np.asarray(pcd1.normals) * normals_weight
velocities = velocities * velocities_weight

features = np.concatenate((points, colors, velocities, normals), axis=1)
#features = np.concatenate((points, colors), axis=1)

db = DBSCAN(eps=2.5, min_samples=4, metric='euclidean', algorithm='auto') #eps 5 is too much
db.fit(features)
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

clustered_pcd1 = o3d.geometry.PointCloud()
clustered_pcd1.points = o3d.utility.Vector3dVector(points)
clustered_pcd1.colors = o3d.utility.Vector3dVector(labels_colors)
print("downsampled 0.01 voxel and sklearn sbcanned clustered pointcloud")
o3d.visualization.draw_geometries([clustered_pcd1])