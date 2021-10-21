import cv2 as cv
import fnmatch
import numpy as np
import os
from PIL import Image, ImageEnhance
import sys


def main(filepath):
    matches = []
    for root, dirnames, filenames in os.walk(filepath):
        for filename in fnmatch.filter(filenames, '*.png'):
            filepath = os.path.join(root, filename)
            matches.append(filepath)
            contrast_enhancement(filepath)


def contrast_enhancement(filepath):
    try:
        img = cv.imread(filepath, 0)
        equ = cv.equalizeHist(img)
        print(filepath)
        res = np.hstack((img, equ))
        cv.imwrite(filepath, equ)
        # equ = cv.equalizeHist(img)
        # res = np.hstack((img, equ))
        # print(filepath.replace("ccmlo", "ccmlo_enhanced"))
        # cv.imshow(equ)
       ## cv.imwrite(filepath.replace("ccmlo", "ccmlo_enhanced"), res)
    except:
        err = sys.exc_info()[0]
        print(str(err))

# filepath: D:\dataset\manifest-ZkhPvrLo5216730872708713142
main('C:\\Users\\andre\\OneDrive\\Desktop\\ccmlo')