import pydicom
import fnmatch
import os
import cv2 as cv


def main(filepath):
    matches = []
    for root, dirnames, filenames in os.walk(filepath):
        for filename in fnmatch.filter(filenames, '*.dcm'):
            filepath = os.path.join(root, filename)
            matches.append(filepath)
            convert_to_png(filepath)


def convert_to_png(filepath):
    ds = pydicom.dcmread(filepath)
    shape = ds.pixel_array.shape
    img = ds.pixel_array # get image array
    cv.imwrite(filepath.replace('.dcm','-original.png'),img)

# filepath: D:\dataset\manifest-ZkhPvrLo5216730872708713142
main(os.getenv('dataset_path'))