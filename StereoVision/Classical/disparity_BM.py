
import numpy as np
import cv2
from matplotlib import pyplot as plt

import scipy.ndimage
import os
import shutil
import scipy.signal
import time

print('loading images...')

left_np = cv2.imread('2p_left.png', cv2.IMREAD_ANYDEPTH)
right_np = cv2.imread('2p_right.png', cv2.IMREAD_ANYDEPTH)
left_np = np.round((left_np - left_np.min()) / (left_np.max() - left_np.min()) * 256).astype(np.uint8);
right_np = np.round((right_np - right_np.min()) / (right_np.max() - right_np.min()) * 256).astype(np.uint8);

# SAD window size should be between 5..255
block_size = 11

min_disp = 0
num_disp = 64 - min_disp
uniquenessRatio = 10
speckle_range = 50
speckle_win_size = 100

left_matcher = cv2.StereoBM_create(numDisparities = num_disp, blockSize = block_size)
left_matcher.setUniquenessRatio(uniquenessRatio)
left_matcher.setMinDisparity(min_disp)
left_matcher.setSpeckleRange(speckle_range)
left_matcher.setSpeckleWindowSize(speckle_win_size)

# disparity = stereo.compute(imgL,imgR)
# disparity = left_matcher.compute(imgL, imgR).astype(np.float32) / 16.0

right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

left_disp = left_matcher.compute(left_np, right_np)
right_disp = right_matcher.compute(right_np,left_np)


wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
wls_filter.setLambda(8000)
wls_filter.setSigmaColor(1.2)
wls_filter.setDepthDiscontinuityRadius(7) # Normal value = 7
wls_filter.setLRCthresh(24)
disparity = np.zeros(left_disp.shape)
# disparity = wls_filter.filter(left_disp,left_np, disparity,right_disp,(0, 0, left_disp.shape[1], left_disp.shape[0]),right_np)
disparity = wls_filter.filter(left_disp, left_np, disparity, right_disp)
confidence_np = wls_filter.getConfidenceMap()

# normalising disparities for saving/display
disparity_norm = disparity.astype(np.float32) / 16
left_disp_norm = left_disp.astype(np.float32) / 16

plt.subplot(211)
plt.imshow(left_disp_norm)
plt.colorbar()
plt.subplot(212)
plt.imshow(disparity_norm)
plt.colorbar()
plt.show()