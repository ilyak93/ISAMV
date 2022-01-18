import os
from PIL import Image
import io
import numpy as np

def main():
    left_path = "G:/Vista_project/left/"
    right_path = "G:/Vista_project/right/"
    path = "G:/Vista_project/"
    files = os.listdir(path)[2:]
    files = [f for f in files if len(f.split('.'))>1 and f.split('.')[1] == "bin"]
    left = [f for f in files if "tc1" in f and "ts" not in f]
    right = [f for f in files if "tc2" in f and "ts" not in f]
    left = [left[0]]
    right = [right[0]]
    img_size = 640*512*2
    name = 0
    for l, r in zip(left,right):
        with open(path+l, "rb") as left_bytes_stream, open(path+r, "rb") as right_bytes_stream:
            while(True):
                left_img_bytes = left_bytes_stream.read(img_size)
                righ_img_bytes = right_bytes_stream.read(img_size)
                if len(left_img_bytes) != img_size or len(righ_img_bytes) != img_size:
                    break
                left_stream = io.BytesIO(left_img_bytes)
                bytes = left_stream.getbuffer()
                left_img_np = np.frombuffer(bytes, dtype='uint16')
                left_img_np_norm = (left_img_np - np.min(left_img_np)) / (np.max(left_img_np) - np.min(left_img_np))
                left_img_np_norm = (left_img_np_norm * 65536).astype(dtype='uint16')
                left_img_np_norm = left_img_np_norm.reshape((512, 640))
                left_img = Image.fromarray(left_img_np_norm)
                #left_img_png = Image.open(y)
                lr = left_img.save(left_path + str(name)+".png", format='PNG')

                right_stream = io.BytesIO(righ_img_bytes)
                bytes = right_stream.getbuffer()
                right_img_np = np.frombuffer(bytes, dtype='uint16')
                right_img_np_norm = (right_img_np - np.min(right_img_np)) / (np.max(right_img_np) - np.min(right_img_np))
                right_img_np_norm = (right_img_np_norm * 65536).astype(dtype='uint16')
                right_img_np_norm = right_img_np_norm.reshape((512, 640))
                right_img = Image.fromarray(right_img_np_norm)
                # left_img_png = Image.open(y)
                r1 = right_img.save(right_path + str(name) + ".png", format='PNG')

                name = name + 1






if __name__ == "__main__":
    main()
