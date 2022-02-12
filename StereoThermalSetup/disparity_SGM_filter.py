import numpy as np
import cv2
from matplotlib import pyplot as plt

import scipy.ndimage
import os
import shutil
import scipy.signal
import time

print('loading images...')
imgL = cv2.imread('2p_left.png', cv2.IMREAD_ANYDEPTH)
imgR = cv2.imread('2p_right.png', cv2.IMREAD_ANYDEPTH)
imgL = np.round((imgL - imgL.min()) / (imgL.max() - imgL.min()) * 256).astype(np.uint8);
imgR = np.round((imgR - imgR.min()) / (imgR.max() - imgR.min()) * 256).astype(np.uint8);


# disparity range tuning
# https://docs.opencv.org/trunk/d2/d85/classcv_1_1StereoSGBM.html
window_size = 3
min_disp = 0
num_disp = 64 - min_disp
left_matcher  = cv2.StereoSGBM_create(
    minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = 11,
    P1 = 8 * 3 * window_size**2,
    P2 = 32 * 3 * window_size**2,
    disp12MaxDiff = 3,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32
    )


right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

left_disp = left_matcher.compute(imgL, imgR)
right_disp = right_matcher.compute(imgR, imgL)


wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
wls_filter.setLambda(8000)
wls_filter.setSigmaColor(1.2)
wls_filter.setDepthDiscontinuityRadius(7) # Normal value = 7
wls_filter.setLRCthresh(24)
disparity = np.zeros(left_disp.shape)
# disparity = wls_filter.filter(left_disp,left_np, disparity,right_disp,(0, 0, left_disp.shape[1], left_disp.shape[0]),right_np)
disparity = wls_filter.filter(left_disp, imgL, disparity, right_disp)
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