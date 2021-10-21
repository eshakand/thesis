import cv2 as cv
import fnmatch
import numpy as np
import os
import sys
from PIL import Image


def main(filepath):
    matches = []
    for root, dirnames, filenames in os.walk(filepath):
        for filename in fnmatch.filter(filenames, '*.png'):
            filepath = os.path.join(root, filename)
            matches.append(filepath)
            flood_filling(filepath)


def flood_filling(filepath):
    try:
        img = Image.open(filepath) 
        rotate_img= img.rotate(345)
        rotate_img.save(filepath.replace('.png', '-45-degree-rotated.png'))
    except:
        err = sys.exc_info()[0]
        print(str(err))


# filepath: D:\dataset\manifest-ZkhPvrLo5216730872708713142
main('C:/Users/andre/OneDrive/Documents/Thesis Project/thesis/mass training/test/rotate')