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
separate_quartet_path = "F:/Vista_project2/12_need_to_be_filtered/sync/"
united_quartet_path = "F:/Vista_project2/12_need_to_be_filtered/sync_presentation/"

#sorting
separate_quartet_files = os.listdir(separate_quartet_path)
separate_quartet_files = winsort(separate_quartet_files)

united_quartet_files = os.listdir(united_quartet_path)
united_quartet_files = winsort(united_quartet_files)

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
for i in range(start_idx + step, united_quartet_files_len, step):
    cur_depth_image_path = separate_quartet_path + separate_quartet_files[i + depth_idx]
    cur_im_depth = cv2.imread(cur_depth_image_path, cv2.IMREAD_UNCHANGED)

    if (cur_im_depth == prev_im_depth).all():
        if start == False:
            start_indices.append(prev_idx)
            start = True
    else:
        prev_im_depth = cur_im_depth
        prev_idx = i
        if start == True:
            end_indices.append(i-1)
            start = False
# special case if last sequence ends without cur_im_depth == prev_im_depth
# for any of the last frames
if start == True:
    end_indices.append(united_quartet_files_len)

#ptints the number of elemets it will delete
sm = sum([end - start + 1 for start, end in zip(start_indices, end_indices)])
print(sm)

# iterate over the quartets and deletes all frames
# from each start and end index found previously
# from seprate quartets and united into one image
for i in range(len(start_indices)):
    start_index_to_delete = start_indices[i]
    end_index_to_delete = end_indices[i]
    for j in range(start_index_to_delete, end_index_to_delete+1):
        os.remove(separate_quartet_path + separate_quartet_files[j])
        if j % step == 0:
            os.remove(united_quartet_path + united_quartet_files[int(j / step)])
