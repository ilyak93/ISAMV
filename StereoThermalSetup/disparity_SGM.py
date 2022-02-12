# skeleton source https://github.com/opencv/opencv/blob/master/samples/python/stereo_match.py
import numpy as np
import cv2
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
imgL = cv2.imread('2p_left.png', cv2.IMREAD_ANYDEPTH)
imgR = cv2.imread('2p_right.png', cv2.IMREAD_ANYDEPTH)
imgL = np.round((imgL - imgL.min()) / (imgL.max() - imgL.min()) * 256).astype(np.uint8);
imgR = np.round((imgR - imgR.min()) / (imgR.max() - imgR.min()) * 256).astype(np.uint8);


# disparity range tuning
# https://docs.opencv.org/trunk/d2/d85/classcv_1_1StereoSGBM.html
window_size = 3
min_disp = 1
num_disp = 64 - min_disp
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = 5,
    P1 = 8 * 3 * window_size**2,
    P2 = 32 * 3 * window_size**2,
    disp12MaxDiff = 3,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32
    )

print('computing disparity...')
# disparity = stereo.compute(imgL, imgR)
disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
plt.imshow(disparity, 'gray')
plt.show()

disparity[disparity>0] = 64 - disparity[disparity>0]


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
                [0,-1, 0,  h/2.0],
                [0, 0, 0, -focal_length],
                [0, 0, 1, 0]])

points_3D = cv2.reprojectImageTo3D(disparity, Q)
points_3D[:,:,1] = points_3D[:,:,1] * (-1)
points_3D[:,:,0] = points_3D[:,:,0] * (-1)


#points_3D[:,:,1] = points_3D[:,:,1] / -1

colors = cv2.cvtColor(imgL,cv2.COLOR_GRAY2RGB)
mask_map = disparity > 0
output_points = points_3D[mask_map]
output_colors = colors[mask_map]

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
pcd = o3d.io.read_point_cloud("./3dmap.ply")
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

#cv2.imshow('Left Image', image_left)
#cv2.imshow('Right Image', image_right)
#cv2.imshow('Disparity Map', (disparity_map - min_disp) / num_disp)
#cv2.waitKey()
#cv2.destroyAllWindows()