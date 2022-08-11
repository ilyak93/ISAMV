import numpy as np
import cv2 as cv
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib



if __name__ == '__main__':
    img = cv.imread('C:/dataset/tracking_dataset1/12color.png')

    colors_dict = {}
    for name, hex in matplotlib.colors.cnames.items():
        colors_dict[name] = matplotlib.colors.to_rgb(hex)
    colors_dict.pop("black")
    num_of_colors = len(colors_dict);
    my_colors_dict = dict()

    for key, color in enumerate(colors_dict.values()):
        my_colors_dict[key] = [t for t in color]

    # filter to reduce noise
    img = cv.medianBlur(img, 3)

    # flatten the image
    flat_image = img.reshape((-1,3))
    flat_image = np.float32(flat_image)

    # meanshift
    bandwidth = estimate_bandwidth(flat_image, n_samples=3000, quantile=.01)
    ms = MeanShift(bandwidth=bandwidth, max_iter=800, bin_seeding=True, n_jobs=-1)
    ms.fit(flat_image)
    labeled=ms.labels_


    # get number of segments
    (segments, counts) = np.unique(labeled, return_counts=True)
    print('Number of segments: ', segments.shape[0])
    _, indices = counts.sort()
    frequencies = np.asarray((segments, counts)).T

    print(frequencies)

    # get the average color of each segment
    total = np.zeros((segments.shape[0], 3), dtype=float)
    count = np.zeros(total.shape, dtype=float)
    for i, label in enumerate(labeled):
        total[label] = total[label] + flat_image[i]
        count[label] += 1
    avg = total/count
    avg = np.uint8(avg)

    # cast the labeled image into the corresponding average color
    res = avg[labeled]
    result = res.reshape((img.shape))

    # show the result
    cv.imshow('result',result)
    cv.waitKey(0)
    cv.destroyAllWindows()