import copy

import cv2
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


color_frame1 = 'ex1.png'
np_color1 = np.array(Image.open(color_frame1)).astype("uint8")
color_im1 = Image.fromarray(np.array(Image.open(color_frame1)).astype("uint8"))
print("Image mode: ", color_im1.mode)

depth_frame1 = 'ex1_d.png'
np_depth1 = np.array(Image.open(depth_frame1)).astype("uint16")
depth_im1 = Image.fromarray(np.array(Image.open(depth_frame1)).astype("uint16"))
print("Image mode: ", depth_im1.mode)



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
downpcd_normals1 = pcd1.voxel_down_sample(voxel_size=0.02)
downpcd_normals1.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
#o3d.visualization.draw_geometries([pcd])
#o3d.visualization.draw_geometries([downpcd_normals], point_show_normal=True)

'''
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

z_min = points[:,2].min()
z_max = points[:,2].max()
x_min = points[:,0].min()
x_max = points[:,0].max()
y_min = points[:,1].min()
y_max = points[:,1].max()

features = np.concatenate((points, (colors*255).astype(np.uint8)), axis=1)


kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(features)
labels = kmeans.labels_
colors = []
for i in range(labels.shape[0]):
    if labels[i] == 0:
        colors.append([0,255,0])
    elif labels[i] == 1:
        colors.append([0,0,255])
    elif labels[i] == 2:
        colors.append([255,255,0])
    elif labels[i] == 3:
        colors.append([0,255,255])
    elif labels[i] == 4:
        colors.append([128,0,0])
    elif labels[i] == 5:
        colors.append([128,128,0])
    elif labels[i] == 6:
        colors.append([0, 128, 0])
    elif labels[i] == 7:
        colors.append([128,0,128])
    elif labels[i] == 8:
        colors.append([0,128,128])
    else:
        colors.append([0,0,128])

labels_colors = np.asarray(colors)
print()


clustered_pcd = o3d.geometry.PointCloud()
clustered_pcd.points = o3d.utility.Vector3dVector(points)
clustered_pcd.colors = o3d.utility.Vector3dVector(labels_colors)
print("k-means clustered 6 dim data to pointcloud")
o3d.visualization.draw_geometries([clustered_pcd])
'''
'''
img = cv2.imread('ex1.png')[66:622,136:1110]
img_d = cv2.imread('ex1_d.png', cv2.IMREAD_ANYDEPTH)[66:622, 136:1110]

img_d = img_d.reshape(list(img_d.shape)+[1])
img_color_depth = np.concatenate((img, img_d), axis=2)


kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(np.reshape(img_color_depth, [-1, 4]))
labels2 = kmeans.labels_

colors2 = []
for i in range(labels2.shape[0]):
    if labels2[i] == 0:
        colors2.append([0,255,0])
    elif labels2[i] == 1:
        colors2.append([0,0,255])
    elif labels2[i] == 2:
        colors2.append([255,255,0])
    elif labels2[i] == 3:
        colors2.append([0,255,255])
    elif labels2[i] == 4:
        colors2.append([128,0,0])
    elif labels2[i] == 5:
        colors2.append([128,128,0])
    elif labels2[i] == 6:
        colors2.append([0, 128, 0])
    elif labels2[i] == 7:
        colors2.append([128,0,128])
    elif labels2[i] == 8:
        colors2.append([0,128,128])
    else:
        colors2.append([0,0,128])

labels_colors2 = np.asarray(colors2)

rgb = o3d.geometry.Image(labels_colors2.astype(np.uint8).reshape([556,974,3]))
depth = o3d.geometry.Image(img_d.astype(np.uint16))
clustered_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    rgb, depth, depth_scale=1000.0, depth_trunc=5, convert_rgb_to_intensity=False)
clustered_pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(
    clustered_rgbd, intrinsic)
# Flip it, otherwise the pointcloud will be upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

o3d.visualization.draw_geometries([clustered_pcd2])
'''
'''

unnormilized_color_pcd = o3d.geometry.PointCloud()
unnormilized_color_pcd.points = o3d.utility.Vector3dVector(points)
colors = (np.asarray(pcd.colors)*255).astype(np.uint8)
unnormilized_color_pcd.colors = o3d.utility.Vector3dVector(colors)
check = np.asarray(unnormilized_color_pcd.colors)

o3d.visualization.draw_geometries([unnormilized_color_pcd])

downpcd = unnormilized_color_pcd.voxel_down_sample(voxel_size=0.02) # normilized or unormilized doesn't matter
check = np.asarray(downpcd.colors)
print("downsampled 0.01 voxel pointcloud")
o3d.visualization.draw_geometries([downpcd])
labels = downpcd.cluster_dbscan(eps=0.05, min_points=2, print_progress=False)
np_labels = np.asarray(labels)
unique_np_labels = np.unique(np_labels).shape

print("number of unique clusters of downsampled pc is"  + str(unique_np_labels))

points = np.asarray(downpcd.points)

colors = []
for i in range(np_labels.shape[0]):
    if np_labels[i] == 0:
        colors.append([0,255,0])
    elif np_labels[i] == 1:
        colors.append([0,0,255])
    elif np_labels[i] == 2:
        colors.append([255,255,0])
    elif np_labels[i] == 3:
        colors.append([0,255,255])
    elif np_labels[i] == 4:
        colors.append([128,0,0])
    elif np_labels[i] == 5:
        colors.append([128,128,0])
    elif np_labels[i] == 6:
        colors.append([0, 128, 0])
    elif np_labels[i] == 7:
        colors.append([128,0,128])
    elif np_labels[i] == 8:
        colors.append([0,128,128])
    else:
        colors.append([0,0,128])

labels_colors = np.asarray(colors)
print()


clustered_pcd = o3d.geometry.PointCloud()
clustered_pcd.points = o3d.utility.Vector3dVector(points)
clustered_pcd.colors = o3d.utility.Vector3dVector(labels_colors)
print("downsampled 0.01 voxel and o3d sbcanned clsutered pointcloud")
o3d.visualization.draw_geometries([clustered_pcd])
'''

downpcd1 = pcd1.voxel_down_sample(voxel_size=0.02)
downpcd1.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

points_weight = 10
color_weight = 50 # 50
normals_weight = 5 # 5
points = np.asarray(downpcd1.points) * points_weight
colors = np.asarray(downpcd1.colors) * color_weight
normals = np.asarray(downpcd1.normals) * normals_weight

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

db = DBSCAN(eps=2, min_samples=2, metric='euclidean', algorithm='auto') #eps 5 is too much
db.fit(features)
labels = db.labels_
colors = []
for i in range(labels.shape[0]):
    if labels[i] == -1:
        colors.append([0,0,0])
    elif labels[i] % 10 == 0:
        colors.append([0,255,0])
    elif labels[i] % 10 == 1:
        colors.append([0,0,255])
    elif labels[i] % 10 == 2:
        colors.append([255,255,0])
    elif labels[i] % 10 == 3:
        colors.append([0,255,255])
    elif labels[i] % 10 == 4:
        colors.append([128,0,0])
    elif labels[i] % 10 == 5:
        colors.append([128,128,0])
    elif labels[i] % 10 == 6:
        colors.append([0, 128, 0])
    elif labels[i] % 10 == 7:
        colors.append([128,0,128])
    elif labels[i] % 10 == 8:
        colors.append([0,128,128])
    else:
        colors.append([0,0,128])

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

'''
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

features = np.concatenate((points, colors), axis=1)


kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(features)
labels = kmeans.labels_
colors = []
for i in range(labels.shape[0]):
    if labels[i] == 0:
        colors.append([0,255,0])
    elif labels[i] == 1:
        colors.append([0,0,255])
    elif labels[i] == 2:
        colors.append([255,255,0])
    elif labels[i] == 3:
        colors.append([0,255,255])
    elif labels[i] == 4:
        colors.append([128,0,0])
    elif labels[i] == 5:
        colors.append([128,128,0])
    elif labels[i] == 6:
        colors.append([0, 128, 0])
    elif labels[i] == 7:
        colors.append([128,0,128])
    elif labels[i] == 8:
        colors.append([0,128,128])
    else:
        colors.append([0,0,128])

labels_colors = np.asarray(colors)
print()


clustered_pcd = o3d.geometry.PointCloud()
clustered_pcd.points = o3d.utility.Vector3dVector(points)
clustered_pcd.colors = o3d.utility.Vector3dVector(labels_colors)

o3d.visualization.draw_geometries([clustered_pcd])
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

pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image1, intrinsic)
# Flip it, otherwise the pointcloud will be upside down
pcd1.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
print("original pointcloud")
pcd1 = pcd1.voxel_down_sample(voxel_size=0.02)
pcd1.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
#o3d.visualization.draw_geometries([pcd1])
#o3d.visualization.draw_geometries([pcd1], point_show_normal=True)


color_raw2 = o3d.io.read_image("ex2.png")
depth_raw2 = o3d.io.read_image("ex2_d.png")


rgbd_image2 = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw2, depth_raw2, depth_scale=1000.0, depth_trunc=5, convert_rgb_to_intensity=False)
print(rgbd_image2)

pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image2, intrinsic)

downpcd2 = pcd2.voxel_down_sample(voxel_size=0.02)


dists = pcd1.compute_point_cloud_distance(downpcd2)

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

dd, ii = my_compute_point_cloud_distance(np.asarray(pcd1.points), downpcd2)

velocities = np.asarray(downpcd2.points)[ii.astype(int),:] - np.asarray(pcd1.points)[:,:]

vz_min = velocities[:,2].min()
vz_max = velocities[:,2].max()
vx_min = velocities[:,0].min()
vx_max = velocities[:,0].max()
vy_min = velocities[:,1].min()
vy_max = velocities[:,1].max()


color_raw1 = o3d.io.read_image("ex1.png")
depth_raw1 = o3d.io.read_image("ex1_d.png")

rgbd_image1 = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw1, depth_raw1, depth_scale=1000.0, depth_trunc=5, convert_rgb_to_intensity=False)
print(rgbd_image1)

pcd3 = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image1, intrinsic)
# Flip it, otherwise the pointcloud will be upside down
pcd3.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
print("original pointcloud")
pcd3 = pcd3.voxel_down_sample(voxel_size=0.02)
pcd3.normals = o3d.utility.Vector3dVector(velocities)
#o3d.visualization.draw_geometries([pcd1])
o3d.visualization.draw_geometries([pcd3], point_show_normal=True)


pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image2, intrinsic)
# Flip it, otherwise the pointcloud will be upside down
pcd2.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
print("original pointcloud")
pcd2 = pcd2.voxel_down_sample(voxel_size=0.02)

pcd2.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
#o3d.visualization.draw_geometries([pcd1], point_show_normal=True)

points_weight = 10
color_weight = 50 # 50
normals_weight = 5 # 5
velocities_weight = 20

points = np.asarray(pcd1.points) * points_weight
colors = np.asarray(pcd1.colors) * color_weight
normals = np.asarray(pcd1.normals) * normals_weight
velocities = velocities * velocities_weight

features = np.concatenate((points, colors, velocities, normals), axis=1)
#features = np.concatenate((points, colors), axis=1)

db = DBSCAN(eps=2, min_samples=2, metric='euclidean', algorithm='auto') #eps 5 is too much
db.fit(features)
labels = db.labels_
colors = []
for i in range(labels.shape[0]):
    if labels[i] == -1:
        colors.append([0,0,0])
    elif labels[i] % 10 == 0:
        colors.append([0,255,0])
    elif labels[i] % 10 == 1:
        colors.append([0,0,255])
    elif labels[i] % 10 == 2:
        colors.append([255,255,0])
    elif labels[i] % 10 == 3:
        colors.append([0,255,255])
    elif labels[i] % 10 == 4:
        colors.append([128,0,0])
    elif labels[i] % 10 == 5:
        colors.append([128,128,0])
    elif labels[i] % 10 == 6:
        colors.append([0, 128, 0])
    elif labels[i] % 10 == 7:
        colors.append([128,0,128])
    elif labels[i] % 10 == 8:
        colors.append([0,128,128])
    else:
        colors.append([0,0,128])

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


'''
def find_closest_vec(point, pc_vecs, best):
    dist = np.inf
    color_diff = np.inf
    idx = -1
    for cur_idx, xyz_rgb in enumerate(pc_vecs):
        cur_dist = np.sqrt(np.sum((point[0:2] - xyz_rgb[0:2]) ** 2))
        cur_color_diff = np.sum(point[3:5] - xyz_rgb[3:5])
        if cur_dist + cur_color_diff < dist + color_diff and np.all(cur_idx != best) and cur_dist > 0 :
                idx = cur_idx
                dist = cur_dist
                color_diff = cur_color_diff
    return pc_vecs[idx], idx

def find_k_closests(src_vectors, dest_vectors, k):
    src_closest = np.zeros((src_vectors.shape[0], k)) - 1
    dest_closest = np.zeros((dest_vectors.shape[0], k)) - 1

    for cur_vec_idx, src_vec in enumerate(src_vectors):
        src_best = np.zeros(k) - 1
        for nn in range(k):
            _, idx = find_closest_vec(src_vec, src_vectors, src_best)
            src_best[nn] = idx
            src_closest[cur_vec_idx, nn] = idx
    for cur_vec_idx, dest_vec in enumerate(dest_vectors):
        cur_i = 0
        dest_best = np.zeros(k) - 1
        for nn in range(k):
            _, idx = find_closest_vec(dest_vec, dest_vectors, dest_best)
            dest_best[nn] = idx
            dest_closest[cur_vec_idx, cur_i] = idx

    return src_closest, dest_closest

def pointcloud_velocity_extraction(src_pc, dest_pc, k=3):
    src_points = np.asarray(src_pc.points)
    src_colors = np.asarray(src_pc.colors)

    dest_points = np.asarray(dest_pc.points)
    dest_colors = np.asarray(dest_pc.colors)

    src_vectors = np.concatenate((src_points, src_colors), axis=1)
    dest_vectors = np.concatenate((dest_points, dest_colors), axis=1)

    velocities = np.zeros_like(src_vectors)

    src_closest, dest_closest = find_k_closests(src_vectors, dest_vectors, k)

    best_dist = np.inf

    for src_idx, src_point in enumerate(src_points):
        for dest_idx, dest_point in enumerate(dest_points):
            cur_dist = 0.0
            closest_to_src = src_closest[src_idx]
            closest_to_dest = dest_closest[dest_idx]
            cur_dist += np.sqrt((src_point - dest_point) ** 2)
            distances = scipy.spatial.distance.cdist(closest_to_src, closest_to_dest)
            for l in range(k):
                min_e, min_idx = distances.min()
                cur_dist += np.sqrt((src_closest[min_idx[0]] - dest_closest[min_idx[1]]) ** 2)
                np.delete(distances, min_idx[0], 0)
                np.delete(distances, min_idx[1], 1)

            if cur_dist < best_dist:
                velocities[src_idx, :] = dest_point[2] - src_point[2]
                best_dist = cur_dist

    return velocities

velocity = pointcloud_velocity_extraction(downpcd_normals1, downpcd_normals)
'''

'''
trans_init = np.asarray([[1., 0., 0., 0.],
       [0., 1., 0., 0.],
       [0., 0., 1., 0.],
       [0., 0., 0., 1.]])

threshold = 0.02
reg_p2p = o3d.pipelines.registration.registration_icp(
    pcd1, pcd, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
print(reg_p2p)
print("Transformation is:")
print(reg_p2p.transformation)

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


source_temp = copy.deepcopy(pcd1)
target_temp = copy.deepcopy(pcd)
source_temp.paint_uniform_color([1, 0.706, 0])
target_temp.paint_uniform_color([0, 0.651, 0.929])
o3d.visualization.draw_geometries([source_temp, target_temp])

draw_registration_result(pcd1, pcd, reg_p2p.transformation)
'''


