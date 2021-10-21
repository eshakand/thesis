import cv2 as cv
import fnmatch
import os
import sys


def main(filepath):
    matches = []
    for root, dirnames, filenames in os.walk(filepath):
        for filename in fnmatch.filter(filenames, '*-original.png'):
            filepath = os.path.join(root, filename)
            matches.append(filepath)
            resize(filepath)


def resize(filepath):
    try:
        img = cv.imread(filepath, cv.IMREAD_UNCHANGED)
        width = 3000
        height = 4000
        dim = (width, height)
        resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)
        cv.imwrite(filepath.replace('-original.png', '-resized.png'), resized)
    except:
        err = sys.exc_info()[0]
        print(str(err))


# file path: D:\dataset\manifest-ZkhPvrLo5216730872708713142
main(os.getenv('dataset_path'))