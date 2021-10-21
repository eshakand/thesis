import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
from PIL import Image
import os
import fnmatch

from shutil import copy, copyfile


def main(filepath):
    matches = []
    for root, dirnames, filenames in os.walk(filepath):
        for filename in fnmatch.filter(filenames, '*MALIGNANT*.png'):
            filepath = os.path.join(root, filename)
            copy(filepath, "C:/Users/andre/OneDrive/Documents/roi training/malignant/")
            matches.append(filepath)

    print(len(matches))
main('C:/Users/andre/Downloads/CBIS_DDSM/png_images/calc_case_description_train_set') 


