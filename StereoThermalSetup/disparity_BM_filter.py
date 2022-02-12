import numpy as np
import cv2
from matplotlib import pyplot as plt

print('loading images...')
imgL = cv2.imread('2p_left.png', cv2.IMREAD_ANYDEPTH)
imgR = cv2.imread('2p_right.png', cv2.IMREAD_ANYDEPTH)
imgL = np.round((imgL - imgL.min()) / (imgL.max() - imgL.min()) * 256).astype(np.uint8);
imgR = np.round((imgR - imgR.min()) / (imgR.max() - imgR.min()) * 256).astype(np.uint8);


# SAD window size should be between 5..255
block_size = 11

min_disp = 0
num_disp = 64 - min_disp
uniquenessRatio = 10


stereo = cv2.StereoBM_create(numDisparities = num_disp, blockSize = block_size)
stereo.setUniquenessRatio(uniquenessRatio)


# disparity = stereo.compute(imgL,imgR)
disparity = stereo.compute(imgL, imgR).astype(np.float32)

print("plotting disparity")
plt.imshow(disparity,'gray')
plt.show()