
import cv2 as cv
import fnmatch
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
from PIL import Image

img = mpimg.imread("C:/Users/andre/OneDrive/Documents/Thesis Project/thesis/image pre-processing/test/benign/p_0032_right_cc.png") 
subimage = [[0 for i in range(256)] for j in range(256)]
for i in range(0, len(img), 256):
    for x in range(0, len(img[1]), 256):
        for y in range(i,i+256,1):
            for z in range(x,x+256,1): 
                subimage[y%256][z%256]=img[z][y]
        imgplot = plt.imshow(subimage,cmap="gray")
        plt.show()
        print(i)
        print(x)
        


#matches = []
#for root, dirnames, filenames in os.walk("C:\Users\andre\OneDrive\Documents\Thesis Project\thesis\image pre-processing\test\benign\p_0032_right_cc.png"):
 #   for filename in fnmatch.filter(filenames, '*.png'):
  #      filepath = os.path.join(root, filename)
   #     matches.append(filepath)
    #    img = Image.open(filepath) 
     #   imgplot = plt.imshow(img)
      #  plt.show()