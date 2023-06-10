import matplotlib.pyplot as plt

from skimage import data
from skimage import transform
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, plot_matches, SIFT
import cv2
import numpy as np

path = "E:/PycharmProjects/ISAMV/SIFT_exp/"
img1 = cv2.imread(path+"0left.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(path+"0right.png", cv2.IMREAD_GRAYSCALE)


descriptor_extractor = SIFT()

descriptor_extractor.detect_and_extract(img1)
keypoints1 = descriptor_extractor.keypoints
descriptors1 = descriptor_extractor.descriptors

descriptor_extractor.detect_and_extract(img2)
keypoints2 = descriptor_extractor.keypoints
descriptors2 = descriptor_extractor.descriptors



matches12 = match_descriptors(descriptors1, descriptors2, max_ratio=1,
                              cross_check=True)

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(11, 8))

plt.gray()

plot_matches(ax[0, 0], img1, img2, keypoints1, keypoints2, matches12)
ax[0, 0].axis('off')
ax[0, 0].set_title("Original Image vs. Flipped Image\n"
                   "(all keypoints and matches)")

matches12_par = matches12[abs(keypoints1[matches12[:, 0]][:, 0] - keypoints2[matches12[:, 1]][:, 0]) < 10]

plot_matches(ax[0, 1], img1, img2, keypoints1, keypoints2, matches12_par[::5],
             only_matches=True)
ax[0, 1].axis('off')
ax[0, 1].set_title("Original Image vs. Flipped Image\n"
                   "(subset of matches for visibility)")


plt.tight_layout()
plt.show()

# create data for BA from SIFT matches

point2d_first_view = keypoints1[matches12_par[:, 0]]
point2d_second_view = keypoints2[matches12_par[:, 1]]

points_2d = np.concatenate((point2d_first_view, point2d_second_view),
                           axis=0).astype(np.uint16)
camera_indices = np.concatenate((np.zeros(matches12_par.shape[0]),
                                 np.zeros(matches12_par.shape[0]) + 1),
                                axis=0).astype(np.uint16)
point_indices = np.concatenate((np.array(list(range(matches12_par.shape[0]))),
                                np.array(list(range(matches12_par.shape[0])))),
                               axis=0).astype(np.uint16)


np.savez("points_2d.npz", points_2d)
np.savez("camera_indices.npz", camera_indices)
np.savez("point_indices.npz", point_indices)
