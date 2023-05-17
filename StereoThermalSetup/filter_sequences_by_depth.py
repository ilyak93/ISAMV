import os
import cv2

from ctypes import wintypes, windll
from functools import cmp_to_key

def winsort(data):
    _StrCmpLogicalW = windll.Shlwapi.StrCmpLogicalW
    _StrCmpLogicalW.argtypes = [wintypes.LPWSTR, wintypes.LPWSTR]
    _StrCmpLogicalW.restype  = wintypes.INT

    cmp_fnc = lambda psz1, psz2: _StrCmpLogicalW(psz1, psz2)
    return sorted(data, key=cmp_to_key(cmp_fnc))

separate_quartet_path = "F:/Vista_project2/4_need_to_be_filtered/sync_copy/"
united_quartet_path = "F:/Vista_project2/4_need_to_be_filtered/sync_presentation_copy/"

separate_quartet_files = os.listdir(separate_quartet_path)
separate_quartet_files = winsort(separate_quartet_files)

united_quartet_files = os.listdir(united_quartet_path)
united_quartet_files = winsort(united_quartet_files)

united_quartet_files_len = len(separate_quartet_files)

start_indices = list()
end_indices = list()
start = False
start_idx = 0
depth_idx = 3
prev_depth_image_path = separate_quartet_path + separate_quartet_files[start_idx + depth_idx]
prev_im_depth = cv2.imread(prev_depth_image_path, cv2.IMREAD_UNCHANGED)
prev_idx = 0
step = 4

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

if start == True:
    end_indices.append(united_quartet_files_len)

sm = sum([end - start + 1 for start, end in zip(start_indices, end_indices)])

print(sm)

for i in range(len(start_indices)):
    start_index_to_delete = start_indices[i]
    end_index_to_delete = end_indices[i]
    for j in range(start_index_to_delete, end_index_to_delete+1):
        os.remove(separate_quartet_path + separate_quartet_files[j])
        if j % step == 0:
            os.remove(united_quartet_path + united_quartet_files[int(j / step)])
