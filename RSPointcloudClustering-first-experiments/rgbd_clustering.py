

import cv2
import numpy as np
import scipy

from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering, OPTICS, MeanShift, SpectralClustering
from matplotlib import pyplot as plt



from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt



img = cv2.imread('ex1.png')[66:622,136:1110]

# Compute DBSCAN

n = 0
while(n<1):
    img = cv2.pyrDown(img)
    n = n+1
rows, cols, ch = img.shape

db = DBSCAN(eps=2, min_samples=300, metric='euclidean', algorithm='auto')
db.fit(np.reshape(img, [-1, 3]))
labels = db.labels_
clusters_count = np.unique(db.labels_).shape

fig = plt.figure(2)
fig.suptitle('rgb dbscan eps=2 min_samples=300, 1 downsample, '+'clusters: ' + str(clusters_count))

plt.subplot(2, 1, 1)
plt.imshow(img)
plt.axis('off')
plt.subplot(2, 1, 2)
plt.imshow(np.reshape(labels, [rows, cols]))
plt.axis('off')
plt.colorbar()
plt.show()


img_d = cv2.imread('ex1_d.png', cv2.IMREAD_ANYDEPTH)[66:622,136:1110]
n = 0
while(n<1):
    img_d = cv2.pyrDown(img_d)
    n = n+1
rows, cols = img_d.shape

db = DBSCAN(eps=1, min_samples=50, metric='euclidean', algorithm='auto')
db.fit(np.reshape(img_d, [-1, 1]))
labels = db.labels_
clusters_count = np.unique(db.labels_).shape

fig = plt.figure(2)
fig.suptitle('depth dbscan eps=1 min_samples=50, 1 downsample, '+'clusters: ' + str(clusters_count))
plt.subplot(2, 1, 1)
plt.imshow(img_d)
plt.axis('off')
plt.subplot(2, 1, 2)
plt.imshow(np.reshape(labels, [rows, cols]))
plt.axis('off')
plt.colorbar()
plt.show()


#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#gray = gray.reshape(list(gray.shape)+[1])
img_d = img_d.reshape(list(img_d.shape)+[1])
img_color_depth = np.concatenate((img, img_d), axis=2)
n = 0
while(n<1):
    img_color_depth = cv2.pyrDown(img_color_depth)
    n = n+1
rows, cols, ch = img_color_depth.shape

db = DBSCAN(eps=20, min_samples=40, metric='euclidean', algorithm='auto')
db.fit(np.reshape(img_color_depth, [-1, 4]))
labels = db.labels_
clusters_count = np.unique(db.labels_).shape

fig = plt.figure(2)
fig.suptitle('rgb&depth dbscan eps=20 min_samples=40, 1 downsample, '+'clusters: ' + str(clusters_count))
plt.subplot(2, 1, 1)
plt.imshow(img_color_depth)
plt.axis('off')
plt.subplot(2, 1, 2)
plt.imshow(np.reshape(labels, [rows, cols]))
plt.axis('off')
plt.colorbar()
plt.show()




img = cv2.imread('ex1.png')[66:622, 136:1110]

rows, cols, ch = img.shape

kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(np.reshape(img, [-1, 3]))
labels = kmeans.labels_
clusters_count = np.unique(labels).shape

fig = plt.figure(2)
fig.suptitle('rgb k-means, 0 downsamples, ' + 'clusters:'+ str(clusters_count))

plt.subplot(2, 1, 1)
plt.imshow(img)
plt.axis('off')
plt.subplot(2, 1, 2)
plt.imshow(np.reshape(labels, [rows, cols]))
plt.axis('off')
plt.colorbar()
plt.show()


img_d = cv2.imread('ex1_d.png', cv2.IMREAD_ANYDEPTH)[66:622, 136:1110]


rows, cols = img_d.shape

kmeans = KMeans(n_clusters=20, random_state=0)
kmeans.fit(np.reshape(img_d, [-1, 1]))
labels = kmeans.labels_
clusters_count = np.unique(labels).shape

fig = plt.figure(2)
fig.suptitle('depth k-means, 0 downsamples, ' + 'clusters:'+ str(clusters_count))

plt.subplot(2, 1, 1)
plt.imshow(img_d)
plt.axis('off')
plt.subplot(2, 1, 2)
plt.imshow(np.reshape(labels, [rows, cols]))
plt.axis('off')
plt.colorbar()
plt.show()

img_d = img_d.reshape(list(img_d.shape)+[1])
img_color_depth = np.concatenate((img, img_d), axis=2)


kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(np.reshape(img_color_depth, [-1, 4]))
labels = kmeans.labels_
clusters_count = np.unique(labels).shape

fig = plt.figure(2)
fig.suptitle('rgb&depth k-means, 0 downsamples, ' + 'clusters:'+ str(clusters_count))
plt.subplot(2, 1, 1)
plt.imshow(img_color_depth)
plt.axis('off')
plt.subplot(2, 1, 2)
plt.imshow(np.reshape(labels, [rows, cols]))
plt.axis('off')
plt.colorbar()
plt.show()



# #############################################################################
img = cv2.imread('ex1.png')[66:622,136:1110]

n = 0
while(n<2):
    img = cv2.pyrDown(img)
    n = n+1
rows, cols, ch = img.shape

af = AgglomerativeClustering(n_clusters=10)
af.fit(np.reshape(img, [-1, 3]))
labels = af.labels_
clusters_count = np.unique(labels).shape

fig = plt.figure(2)
fig.suptitle('rgb AgglomerativeClustering, 2 downsample,' + 'clusters:'+ str(clusters_count))

plt.subplot(2, 1, 1)
plt.imshow(img)
plt.axis('off')
plt.subplot(2, 1, 2)
plt.imshow(np.reshape(labels, [rows, cols]))
plt.axis('off')
plt.colorbar()
plt.show()


img_d = cv2.imread('ex1_d.png', cv2.IMREAD_ANYDEPTH)[66:622,136:1110]
n = 0
while(n<2):
    img_d = cv2.pyrDown(img_d)
    n = n+1
rows, cols = img_d.shape

af = AgglomerativeClustering(n_clusters=10)
af.fit(np.reshape(img_d, [-1, 1]))
labels = af.labels_
clusters_count = np.unique(labels).shape

fig = plt.figure(2)
fig.suptitle('depth AgglomerativeClustering, 2 downsample,' + 'clusters:'+ str(clusters_count))
plt.subplot(2, 1, 1)
plt.imshow(img_d)
plt.axis('off')
plt.subplot(2, 1, 2)
plt.imshow(np.reshape(labels, [rows, cols]))
plt.axis('off')
plt.colorbar()
plt.show()


#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#gray = gray.reshape(list(gray.shape)+[1])
img_d = img_d.reshape(list(img_d.shape)+[1])
img_color_depth = np.concatenate((img, img_d), axis=2)
n = 0
while(n<2):
    img_color_depth = cv2.pyrDown(img_color_depth)
    n = n+1
rows, cols, ch = img_color_depth.shape

af = AgglomerativeClustering(n_clusters=10)
af.fit(np.reshape(img_color_depth, [-1, 4]))
labels = af.labels_
clusters_count = np.unique(labels).shape

fig = plt.figure(2)
fig.suptitle('rgb&depth AgglomerativeClustering, 2 downsample,' + 'clusters:'+ str(clusters_count))
plt.subplot(2, 1, 1)
plt.imshow(img_color_depth)
plt.axis('off')
plt.subplot(2, 1, 2)
plt.imshow(np.reshape(labels, [rows, cols]))
plt.axis('off')
plt.colorbar()
plt.show()