from ctypes import windll, wintypes
from functools import cmp_to_key

import numpy as np
from numba import njit
import os
import cv2
from PIL import Image

@njit
def quickselect(arr, k):
    low = 0
    high = len(arr) - 1
    while low <= high:
        pivot_idx = np.random.randint(low, high + 1)
        pivot = arr[pivot_idx]
        arr[pivot_idx], arr[high] = arr[high], arr[pivot_idx]

        i = low
        for j in range(low, high):
            if arr[j] < pivot:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
        arr[i], arr[high] = arr[high], arr[i]

        if k == i:
            return pivot
        elif k < i:
            high = i - 1
        else:
            low = i + 1

@njit
def quantile_filter(input_array, filter_size, quantile):
    if filter_size % 2 == 0:
        raise ValueError("Filter size must be an odd number.")
    if not 0 <= quantile <= 1:
        raise ValueError("Quantile value must be between 0 and 1.")

    pad_width = filter_size // 2
    output_array = np.zeros_like(input_array)

    for i in range(input_array.shape[0]):
        for j in range(input_array.shape[1]):
            if input_array[i, j] == 0:
                window = []
                for r in range(-pad_width, pad_width + 1):
                    for c in range(-pad_width, pad_width + 1):
                        ni, nj = i + r, j + c
                        if 0 <= ni < input_array.shape[0] and 0 <= nj < input_array.shape[1]:
                            window.append(input_array[ni, nj])
                window = np.array(window)
                quantile_index = int(len(window) * quantile)
                quantile_value = quickselect(window, quantile_index)
                output_array[i, j] = quantile_value
            else:
                output_array[i, j] = input_array[i, j]

    return output_array

def winsort(data):
    _StrCmpLogicalW = windll.Shlwapi.StrCmpLogicalW
    _StrCmpLogicalW.argtypes = [wintypes.LPWSTR, wintypes.LPWSTR]
    _StrCmpLogicalW.restype  = wintypes.INT

    cmp_fnc = lambda psz1, psz2: _StrCmpLogicalW(psz1, psz2)
    return sorted(data, key=cmp_to_key(cmp_fnc))

if __name__ == '__main__':
    gen_path = "J:/Vista_project2/new/new_ds/21_1000/dance_ds/"
    path = "/train/"
    input_path = gen_path + path
    files = os.listdir(input_path)
    def contain_depth(name):
        return "depth" in name
    depth_files = list(filter(contain_depth, files))
    depth_files = winsort(depth_files)
    dest_path = gen_path + "/filled_depths/"
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)

    for depth_file in depth_files:
        depth = cv2.imread(input_path + depth_file, cv2.IMREAD_UNCHANGED)
        for i in range(2):
            depth = quantile_filter(depth, 3, 0.9)

        filled_depth = Image.fromarray(depth, 'I;16')

        # Save the image as a 16-bit PNG file
        filled_depth.save(dest_path + depth_file)




