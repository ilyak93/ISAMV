import os
import cv2

from ctypes import wintypes, windll
from functools import cmp_to_key

# function which sorts a container of strings exactly as windows does
def winsort(data):
    _StrCmpLogicalW = windll.Shlwapi.StrCmpLogicalW
    _StrCmpLogicalW.argtypes = [wintypes.LPWSTR, wintypes.LPWSTR]
    _StrCmpLogicalW.restype  = wintypes.INT

    cmp_fnc = lambda psz1, psz2: _StrCmpLogicalW(psz1, psz2)
    return sorted(data, key=cmp_to_key(cmp_fnc))

#path to files in quartets
separate_quartet_path = "F:/Vista_project2/21_ready/sync/"
#united_quartet_path = "F:/Vista_project2/21_ready/sync_presentation/"

#sorting
separate_quartet_files = os.listdir(separate_quartet_path)
separate_quartet_files = winsort(separate_quartet_files)

#united_quartet_files = os.listdir(united_quartet_path)
#united_quartet_files = winsort(united_quartet_files)

united_quartet_files_len = len(separate_quartet_files)

# start_ indices and end_indices are the stard/end indices of the frames
# for which the depth is the same, i.e freezed
start_indices = list()
end_indices = list()

#start is a flag which helps to control when the current sequence of freezed
# frames starts
start = False
# start idx is used to load the first img manually
start_idx = 0
# depth idx is the position od depth frame in each quartet of frames
# thermal_left, thermal_right, color, depth
depth_idx = 3
color_idx = 2
# loading first image
prev_depth_image_path = separate_quartet_path + separate_quartet_files[start_idx + depth_idx]
prev_im_depth = cv2.imread(prev_depth_image_path, cv2.IMREAD_UNCHANGED)

#prev_idx helps to include the first freezed depth frame
prev_idx = 0

#step is the step to iterate over all quartets
step = 4

# for loop runs from the second depth on all of the depth images
# and compares them sequentially. If two depth frames are the same
# it set the flag start to True, write the starting index of the sequence of freezed frames,
# and continue to iterate until it find the last element of it to write its index
for i in range(2, int(united_quartet_files_len / step)):
    cur_depth_image_path = separate_quartet_path + separate_quartet_files[i * step + depth_idx]
    cur_color_image_path = separate_quartet_path + separate_quartet_files[i * step + color_idx]

    cur_thermal_image_path = separate_quartet_path + separate_quartet_files[i * step]
    if "l" in cur_thermal_image_path:
        left_file_path = cur_thermal_image_path
        right_file_path = separate_quartet_path + separate_quartet_files[i * step + 1]
    else:
        right_file_path = cur_thermal_image_path
        left_file_path = separate_quartet_path + separate_quartet_files[i * step + 1]

    os.rename(cur_color_image_path, separate_quartet_path + str(i) + "color.png")
    os.rename(cur_depth_image_path, separate_quartet_path + str(i) + "depth.png")
    os.rename(left_file_path, separate_quartet_path + str(i) + "left.png")
    os.rename(right_file_path, separate_quartet_path + str(i) + "right.png")
