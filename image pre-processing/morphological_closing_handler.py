import cv2 as cv
import fnmatch
import numpy as np
import os
import sys


def main(filepath):
    matches = []
    for root, dirnames, filenames in os.walk(filepath):
        for filename in fnmatch.filter(filenames, '*-original.png'):
            print(filename)
            filepath = os.path.join(root, filename)
            matches.append(filepath)
            morphological_closing(filepath)


def morphological_closing(filepath):
    try:
        img = cv.imread(filepath,0)
        kernel = np.ones((11,11),np.uint8)
        output = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
        cv.imwrite(filepath.replace('-original.png', '-closed.png'), output)
    except:
        err = sys.exc_info()[0]
        print(str(err))


# path: D:\dataset\manifest-ZkhPvrLo5216730872708713142
main(os.getenv('dataset_path'))