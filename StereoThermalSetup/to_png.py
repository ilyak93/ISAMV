import os
from PIL import Image
import io
import numpy as np

def main():
    left_path = "G:/Vista_project/times_meavrer/left/"
    right_path = "G:/Vista_project/times_meavrer/right/"
    rs_path = "G:/Vista_project/times_meavrer/rs/"
    path = "G:/Vista_project/times_meavrer/"
    files = os.listdir(path)[:-3]
    files = [f for f in files if len(f.split('.'))>1 and f.split('.')[1] == "bin"]
    left = [f for f in files if "tc1" in f and "ts" not in f]
    right = [f for f in files if "tc2" in f and "ts" not in f]
    rs_color = [f for f in files if "color" in f and "ts" not in f]
    rs_depth = [f for f in files if "depth" in f and "ts" not in f]

    left_ts = [f for f in files if "tc1" in f and "ts" in f]
    right_ts = [f for f in files if "tc2" in f and "ts" in f]
    rs_color_ts = [f for f in files if "color" in f and "ts" in f]
    rs_depth_ts = [f for f in files if "depth" in f and "ts" in f]

    img_size = 640*512*2
    ts_size = 8
    name = 0
    prev = 0

    for l, r, lts, rts in zip(left,right, left_ts, right_ts):
        with open(path+l, "rb") as left_bytes_stream,\
                open(path+r, "rb") as right_bytes_stream,\
                open(path + lts, "rb") as left_ts_bytes_stream,\
                open(path + rts, "rb") as right_ts_bytes_stream :

            while(True):
                left_img_bytes = left_bytes_stream.read(img_size)
                right_img_bytes = right_bytes_stream.read(img_size)
                left_img_ts_bytes = left_ts_bytes_stream.read(ts_size)
                right_img_ts_bytes = right_ts_bytes_stream.read(ts_size)
                if len(left_img_bytes) != img_size or \
                    len(right_img_bytes) != img_size or \
                    len(left_img_ts_bytes) != ts_size or \
                    len(right_img_ts_bytes) != ts_size :
                    break
                left_stream = io.BytesIO(left_img_bytes)
                bytes = left_stream.getbuffer()
                left_img_np = np.frombuffer(bytes, dtype='uint16')
                left_img_np_norm = (left_img_np - np.min(left_img_np)) / (np.max(left_img_np) - np.min(left_img_np))
                left_img_np_norm = (left_img_np_norm * 65536).astype(dtype='uint16')
                left_img_np = left_img_np_norm.reshape((512, 640))
                left_img = Image.fromarray(left_img_np)

                left_ts_stream = io.BytesIO(left_img_ts_bytes)
                ts_bytes = left_ts_stream.getbuffer()
                left_img_ts_np = np.frombuffer(ts_bytes, dtype='longlong')
                ts = left_img_ts_np.item()
                cur = ts
                assert prev < cur
                prev = cur
                left_img.save(left_path + str(name)+'_' + str(ts) + ".png", format='PNG')

                right_stream = io.BytesIO(right_img_bytes)
                bytes = right_stream.getbuffer()
                right_img_np = np.frombuffer(bytes, dtype='uint16')
                right_img_np_norm = (right_img_np - np.min(right_img_np)) / (np.max(right_img_np) - np.min(right_img_np))
                right_img_np_norm = (right_img_np_norm * 65536).astype(dtype='uint16')
                right_img_np = right_img_np_norm.reshape((512, 640))
                right_img = Image.fromarray(right_img_np)

                right_ts_stream = io.BytesIO(right_img_ts_bytes)
                ts_bytes = right_ts_stream.getbuffer()
                right_img_ts_np = np.frombuffer(ts_bytes, dtype='longlong')
                ts = right_img_ts_np.item()
                right_img.save(right_path + str(name) + '_' + str(ts) + ".png", format='PNG')

                name = name + 1


    color_img_size = 1280 * 720 * 3
    depth_img_size = 1280 * 720 * 2

    name = 0

    for color, depth, color_ts in zip(rs_color, rs_depth, rs_color_ts):
        with open(path + color, "rb") as color_bytes_stream, \
                open(path + depth, "rb") as depth_bytes_stream, \
                open(path + color_ts, "rb") as color_ts_bytes_stream :
            while (True):
                color_img_bytes = color_bytes_stream.read(color_img_size)
                depth_img_bytes = depth_bytes_stream.read(depth_img_size)
                color_img_ts_bytes = color_ts_bytes_stream.read(ts_size)

                if len(color_img_bytes) != color_img_size or \
                        len(depth_img_bytes) != depth_img_size or \
                        len(color_img_ts_bytes) != ts_size:
                    break

                color_ts_stream = io.BytesIO(color_img_ts_bytes)
                ts_bytes = color_ts_stream.getbuffer()
                color_img_ts_np = np.frombuffer(ts_bytes, dtype='longlong')
                ts = color_img_ts_np.item()

                color_stream = io.BytesIO(color_img_bytes)
                bytes = color_stream.getbuffer()
                color_img_np = np.frombuffer(bytes, dtype='uint8').reshape((720, 1280, 3))
                color_img = Image.fromarray(color_img_np)
                color_img.save(rs_path + str(name) + '_' + str(ts) + "color.png", format='PNG')

                depth_stream = io.BytesIO(depth_img_bytes)
                bytes = depth_stream.getbuffer()
                depth_img_np = np.frombuffer(bytes, dtype='uint16').reshape((720, 1280))
                right_img = Image.fromarray(depth_img_np)
                right_img.save(rs_path + str(name) + '_' + str(ts) + "depth.png", format='PNG')

                name = name + 1






if __name__ == "__main__":
    main()
