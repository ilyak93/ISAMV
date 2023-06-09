import matplotlib.pyplot as plt

from skimage import data
from skimage import transform
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, plot_matches, SIFT
import cv2

path = "/aligned_to_depth_21_ready/"
img1 = cv2.imread(path+"nd.png", cv2.COLOR_BGR2RGB)
img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
img2 = cv2.imread(path+"nt.png", cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread("E:/PycharmProjects/ISAMV/aligned_to_depth_3_ready/"
                  +"nt.png", cv2.IMREAD_GRAYSCALE)

import numpy as np
def median_filter_handmode(image, ksize):
    # Create a Gaussian kernel

    # Apply padding to the image
    pad = ksize // 2
    padded_image = np.zeros((image.shape[0] + pad * 2,
                             image.shape[1] + pad * 2), dtype=np.uint8)

    padded_image[:, :] = np.pad(image[:, :], pad, mode='constant')

    # Convolve the padded image with the Gaussian kernel
    filtered_image = np.zeros_like(padded_image)
    height, width = image.shape

    for i in range(height - ksize + 1):
        for j in range(width - ksize + 1):
            patch = image[i:i + ksize, j:j + ksize]
            med = np.quantile(patch, 0.99)
            filtered_image[i, j] = med


    return filtered_image



# Apply the Gaussian filter
img1 = median_filter_handmode(image=img1, ksize=3)  # Sigma: 1.0


descriptor_extractor = SIFT()

descriptor_extractor.detect_and_extract(img1)
keypoints1 = descriptor_extractor.keypoints
descriptors1 = descriptor_extractor.descriptors

descriptor_extractor.detect_and_extract(img2)
keypoints2 = descriptor_extractor.keypoints
descriptors2 = descriptor_extractor.descriptors

descriptor_extractor.detect_and_extract(img3)
keypoints3 = descriptor_extractor.keypoints
descriptors3 = descriptor_extractor.descriptors

matches12 = match_descriptors(descriptors1, descriptors2, max_ratio=1,
                              cross_check=True)
matches23 = match_descriptors(descriptors2, descriptors3, max_ratio=0.6,
                              cross_check=True)
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(11, 8))

plt.gray()

plot_matches(ax[0, 0], img1, img2, keypoints1, keypoints2, matches12)
ax[0, 0].axis('off')
ax[0, 0].set_title("Original Image vs. Flipped Image\n"
                   "(all keypoints and matches)")

plot_matches(ax[1, 0], img2, img3, keypoints2, keypoints3, matches23)
ax[1, 0].axis('off')
ax[1, 0].set_title("Original Image vs. Transformed Image\n"
                   "(all keypoints and matches)")

matches12_par = matches12[np.sqrt((keypoints1[matches12[:, 0]][:, 0] - keypoints2[matches12[:, 1]][:, 0]) ** 2 + (keypoints1[matches12[:, 0]][:, 1] - keypoints2[matches12[:, 1]][:, 1]) ** 2 ) < 10]

plot_matches(ax[0, 1], img1, img2, keypoints1, keypoints2, matches12_par,
             only_matches=True)
ax[0, 1].axis('off')
ax[0, 1].set_title("Original Image vs. Flipped Image\n"
                   "(subset of matches for visibility)")

plot_matches(ax[1, 1], img2, img3, keypoints2, keypoints3, matches23[::15],
             only_matches=True)
ax[1, 1].axis('off')
ax[1, 1].set_title("Original Image vs. Transformed Image\n"
                   "(subset of matches for visibility)")

plt.tight_layout()
plt.show()