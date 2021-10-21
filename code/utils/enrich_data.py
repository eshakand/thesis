import cv2 as cv
import fnmatch
from matplotlib.image import imread
import numpy as np
import os
from PIL import Image, ImageEnhance
import sys
from matplotlib import pyplot as plt
from scipy import ndimage
import math 

#rotation angle in degree


def main(filepath):
    matches = []
    new_image = np.zeros((9000, 9000, 3))
    width = 0 
    height = 0 
    for root, dirnames, filenames in os.walk(filepath):
        for filename in fnmatch.filter(filenames, '*.png'):
            print(filepath)
            filepath = os.path.join(root, filename)
            matches.append(filepath)
            print(filepath)
            rotate(filepath)

def massage_data(filepath):
        img = Image.open(filepath) 
        if (img.height <= 9000 and img.height >= 0000 and img.width <= 9000 and img.width >= 000):
            temp = Image.open(filepath)
            height= temp.height
            width = temp.width
            image = cv.imread(filepath)

            border = cv.copyMakeBorder(
            image,
            top=math.floor((9000-height)/2),
            bottom=math.floor((9000-height)/2),
            left=math.floor((9000-width)/2),
            right=math.floor((9000-width)/2),
            borderType=cv.BORDER_CONSTANT,
            value=[0, 0, 0]
            )
            cv.imwrite(filepath.replace("*.png", "*-resized.png"), border)

def rotate(filepath):
    image = Image.open(filepath)
    rotatePlus5 = image.rotate(5)
    rotatePlus5.save(filepath.replace(".png", "-rotated+5.png"))
    rotateMinus5 = image.rotate(-5)
    rotateMinus5.save(filepath.replace(".png", "-rotated-5.png"))
    rotatePlus10 = image.rotate(10)
    rotatePlus10.save(filepath.replace(".png", "-rotated+10.png"))
    rotateMinus10 = image.rotate(-10)
    rotateMinus10.save(filepath.replace(".png", "-rotated-10.png"))
    rotatePlus15 = image.rotate(15)
    rotatePlus15.save(filepath.replace(".png", "-rotated+15.png"))
    rotateMinus15 = image.rotate(-15)
    rotateMinus15.save(filepath.replace(".png", "-rotated-15.png"))
    rotatePlus20 = image.rotate(20)
    rotatePlus20.save(filepath.replace(".png", "-rotated+20.png"))
    rotateMinus20 = image.rotate(-20)
    rotateMinus20.save(filepath.replace(".png", "-rotated-20.png"))    
    rotatePlus25 = image.rotate(25)
    rotatePlus25.save(filepath.replace("*.png", "*-rotated+25.png"))
    rotateMinus25 = image.rotate(-25)
    rotateMinus25.save(filepath.replace("*.png", "*-rotated-25.png"))  
    rotatePlus30 = image.rotate(30)
    rotatePlus30.save(filepath.replace("*.png", "*-rotated+30.png"))
    rotateMinus30 = image.rotate(-30)
    rotateMinus30.save(filepath.replace("*.png", "*-rotated-30.png"))      
# filepath: D:\dataset\manifest-ZkhPvrLo52167308727087131p42
main('C:\\Users\\andre\\OneDrive\\Desktop\\cnn_dataset')