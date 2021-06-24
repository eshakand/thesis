import cv2 as cv
import fnmatch
import numpy as np
import os
import sys


def main(filepath):
    matches = []
    for root, dirnames, filenames in os.walk(filepath):
        for filename in fnmatch.filter(filenames, '*-closed.png'):
            filepath = os.path.join(root, filename)
            matches.append(filepath)
            flood_filling(filepath)


def flood_filling(filepath):
    try:
        input_image = cv.imread(filepath)
        im_flood_fill = input_image.copy()
        h, w = input_image.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        im_flood_fill = im_flood_fill.astype("uint8")
        cv.floodFill(im_flood_fill, mask, (0, 0), 255)
        im_flood_fill_inv = cv.bitwise_not(im_flood_fill)
        img_out = im_flood_fill_inv
        cv.imwrite(filepath.replace('-closed.png', '-filled.png'), img_out)
    except:
        err = sys.exc_info()[0]
        print(str(err))


# filepath: D:\dataset\manifest-ZkhPvrLo5216730872708713142
main(os.getenv('dataset_path'))