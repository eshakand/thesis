import cv2 as cv
import fnmatch
import numpy as np
import os
import sys
from PIL import Image

sizeDict = {"<500 pixels by < 500 pixels": 0, 
            "<500 pixels by < 1000 pixels": 0, 
            "<500 pixels by < 1500 pixels": 0, 
            "<500 pixels by < 2000 pixels": 0, 
            "<500 pixels by < 2500 pixels": 0, 
            "<500 pixels by < 3000 pixels": 0, 
            "<500 pixels by < 3500 pixels": 0, 
            "<500 pixels by < 4000 pixels": 0, 
            "<500 pixels by < 4500 pixels": 0, 
            "<500 pixels by < 5000 pixels": 0, 
            "<500 pixels by < 5500 pixels": 0, 
            "<500 pixels by < 6000 pixels": 0, 
            "<500 pixels by < 6500 pixels": 0, 
            "<500 pixels by < 7000 pixels": 0, 
            "<500 pixels by < 7500 pixels": 0, 
            "<1000 pixels by < 500 pixels": 0, 
            "<1000 pixels by < 1000 pixels": 0, 
            "<1000 pixels by < 1500 pixels": 0, 
            "<1000 pixels by < 2000 pixels": 0, 
            "<1000 pixels by < 2500 pixels": 0, 
            "<1000 pixels by < 3000 pixels": 0, 
            "<1000 pixels by < 3500 pixels": 0, 
            "<1000 pixels by < 4000 pixels": 0, 
            "<1000 pixels by < 4500 pixels": 0, 
            "<1000 pixels by < 5000 pixels": 0, 
            "<1000 pixels by < 5500 pixels": 0, 
            "<1000 pixels by < 6000 pixels": 0, 
            "<1000 pixels by < 6500 pixels": 0, 
            "<1000 pixels by < 7000 pixels": 0, 
            "<1000 pixels by < 7500 pixels": 0, 
            "<1500 pixels by < 500 pixels": 0, 
            "<1500 pixels by < 1000 pixels": 0, 
            "<1500 pixels by < 1500 pixels": 0, 
            "<1500 pixels by < 2000 pixels": 0, 
            "<1500 pixels by < 2500 pixels": 0, 
            "<1500 pixels by < 3000 pixels": 0, 
            "<1500 pixels by < 3500 pixels": 0, 
            "<1500 pixels by < 4000 pixels": 0, 
            "<1500 pixels by < 4500 pixels": 0, 
            "<1500 pixels by < 5000 pixels": 0, 
            "<1500 pixels by < 5500 pixels": 0, 
            "<1500 pixels by < 6000 pixels": 0, 
            "<1500 pixels by < 6500 pixels": 0, 
            "<1500 pixels by < 7000 pixels": 0, 
            "<1500 pixels by < 7500 pixels": 0, 
            "<2000 pixels by < 500 pixels": 0, 
            "<2000 pixels by < 1000 pixels": 0, 
            "<2000 pixels by < 1500 pixels": 0, 
            "<2000 pixels by < 2000 pixels": 0, 
            "<2000 pixels by < 2500 pixels": 0, 
            "<1000 pixels by < 3000 pixels": 0, 
            "<2000 pixels by < 3500 pixels": 0, 
            "<2000 pixels by < 4000 pixels": 0, 
            "<2000 pixels by < 4500 pixels": 0, 
            "<2000 pixels by < 5000 pixels": 0, 
            "<2000 pixels by < 5500 pixels": 0, 
            "<2000 pixels by < 6000 pixels": 0, 
            "<2000 pixels by < 6500 pixels": 0, 
            "<2000 pixels by < 7000 pixels": 0, 
            "<2000 pixels by < 7500 pixels": 0, 
            "<2500 pixels by < 500 pixels": 0, 
            "<2500 pixels by < 1000 pixels": 0, 
            "<2500 pixels by < 1500 pixels": 0, 
            "<2500 pixels by < 2000 pixels": 0, 
            "<2500 pixels by < 2500 pixels": 0, 
            "<1500 pixels by < 3000 pixels": 0, 
            "<2500 pixels by < 3500 pixels": 0, 
            "<2500 pixels by < 4000 pixels": 0, 
            "<2500 pixels by < 4500 pixels": 0, 
            "<2500 pixels by < 5000 pixels": 0, 
            "<2500 pixels by < 5500 pixels": 0, 
            "<2500 pixels by < 6000 pixels": 0, 
            "<2500 pixels by < 6500 pixels": 0, 
            "<2500 pixels by < 7000 pixels": 0, 
            "<2500 pixels by < 7500 pixels": 0, 
            "<3000 pixels by < 500 pixels": 0, 
            "<3000 pixels by < 1000 pixels": 0, 
            "<3000 pixels by < 1500 pixels": 0, 
            "<3000 pixels by < 2000 pixels": 0, 
            "<3000 pixels by < 2500 pixels": 0, 
            "<3000 pixels by < 3000 pixels": 0, 
            "<3000 pixels by < 3500 pixels": 0, 
            "<3000 pixels by < 4000 pixels": 0, 
            "<3000 pixels by < 4500 pixels": 0, 
            "<3000 pixels by < 5000 pixels": 0, 
            "<3000 pixels by < 5500 pixels": 0, 
            "<3000 pixels by < 6000 pixels": 0, 
            "<3000 pixels by < 6500 pixels": 0, 
            "<3000 pixels by < 7000 pixels": 0, 
            "<3000 pixels by < 7500 pixels": 0, 
            "<3500 pixels by < 500 pixels": 0, 
            "<3500 pixels by < 1000 pixels": 0, 
            "<3500 pixels by < 1500 pixels": 0, 
            "<3500 pixels by < 2000 pixels": 0, 
            "<3500 pixels by < 2500 pixels": 0, 
            "<3500 pixels by < 3000 pixels": 0, 
            "<3500 pixels by < 3500 pixels": 0, 
            "<3500 pixels by < 4000 pixels": 0, 
            "<3500 pixels by < 4500 pixels": 0, 
            "<3500 pixels by < 5000 pixels": 0, 
            "<3500 pixels by < 5500 pixels": 0, 
            "<3500 pixels by < 6000 pixels": 0, 
            "<3500 pixels by < 6500 pixels": 0, 
            "<3500 pixels by < 7000 pixels": 0, 
            "<3500 pixels by < 7500 pixels": 0, 
            "<4000 pixels by < 500 pixels": 0, 
            "<4000 pixels by < 1000 pixels": 0, 
            "<4000 pixels by < 1500 pixels": 0, 
            "<4000 pixels by < 2000 pixels": 0, 
            "<4000 pixels by < 2500 pixels": 0, 
            "<4000 pixels by < 3000 pixels": 0, 
            "<4000 pixels by < 3500 pixels": 0, 
            "<4000 pixels by < 4000 pixels": 0, 
            "<4000 pixels by < 4500 pixels": 0, 
            "<4000 pixels by < 5000 pixels": 0, 
            "<4000 pixels by < 5500 pixels": 0, 
            "<4000 pixels by < 6000 pixels": 0, 
            "<4000 pixels by < 6500 pixels": 0, 
            "<4000 pixels by < 7000 pixels": 0, 
            "<4000 pixels by < 7500 pixels": 0, 
            "<4500 pixels by < 500 pixels": 0, 
            "<4500 pixels by < 1000 pixels": 0, 
            "<4500 pixels by < 1500 pixels": 0, 
            "<4500 pixels by < 2000 pixels": 0, 
            "<4500 pixels by < 2500 pixels": 0, 
            "<4500 pixels by < 3000 pixels": 0, 
            "<4500 pixels by < 3500 pixels": 0, 
            "<4500 pixels by < 4000 pixels": 0, 
            "<4500 pixels by < 4500 pixels": 0, 
            "<4500 pixels by < 5000 pixels": 0, 
            "<4500 pixels by < 5500 pixels": 0, 
            "<4500 pixels by < 6000 pixels": 0, 
            "<4500 pixels by < 6500 pixels": 0, 
            "<4500 pixels by < 7000 pixels": 0, 
            "<4500 pixels by < 7500 pixels": 0, 
            "<5000 pixels by < 500 pixels": 0, 
            "<5000 pixels by < 1000 pixels": 0, 
            "<5000 pixels by < 1500 pixels": 0, 
            "<5000 pixels by < 2000 pixels": 0, 
            "<5000 pixels by < 2500 pixels": 0, 
            "<5000 pixels by < 3000 pixels": 0, 
            "<5000 pixels by < 3500 pixels": 0, 
            "<5000 pixels by < 4000 pixels": 0, 
            "<5000 pixels by < 4500 pixels": 0, 
            "<5000 pixels by < 5000 pixels": 0, 
            "<5000 pixels by < 5500 pixels": 0, 
            "<5000 pixels by < 6000 pixels": 0, 
            "<5000 pixels by < 6500 pixels": 0, 
            "<5000 pixels by < 7000 pixels": 0, 
            "<5000 pixels by < 7500 pixels": 0, 
            "<5500 pixels by < 500 pixels": 0, 
            "<5500 pixels by < 1000 pixels": 0, 
            "<5500 pixels by < 1500 pixels": 0, 
            "<5500 pixels by < 2000 pixels": 0, 
            "<5500 pixels by < 2500 pixels": 0, 
            "<5500 pixels by < 3000 pixels": 0, 
            "<5500 pixels by < 3500 pixels": 0, 
            "<5500 pixels by < 4000 pixels": 0, 
            "<5500 pixels by < 4500 pixels": 0, 
            "<5500 pixels by < 5000 pixels": 0, 
            "<5500 pixels by < 5500 pixels": 0, 
            "<5500 pixels by < 6000 pixels": 0, 
            "<5500 pixels by < 6500 pixels": 0, 
            "<5500 pixels by < 7000 pixels": 0, 
            "<5500 pixels by < 7500 pixels": 0, 
            "<6000 pixels by < 500 pixels": 0, 
            "<6000 pixels by < 1000 pixels": 0, 
            "<6000 pixels by < 1500 pixels": 0, 
            "<6000 pixels by < 2000 pixels": 0, 
            "<6000 pixels by < 2500 pixels": 0, 
            "<6000 pixels by < 3000 pixels": 0, 
            "<6000 pixels by < 3500 pixels": 0, 
            "<6000 pixels by < 4000 pixels": 0, 
            "<6000 pixels by < 4500 pixels": 0, 
            "<6000 pixels by < 5000 pixels": 0, 
            "<6000 pixels by < 5500 pixels": 0, 
            "<6000 pixels by < 6000 pixels": 0, 
            "<6000 pixels by < 6500 pixels": 0, 
            "<6000 pixels by < 7000 pixels": 0, 
            "<6000 pixels by < 7500 pixels": 0, 
            "<6500 pixels by < 500 pixels": 0, 
            "<6500 pixels by < 1000 pixels": 0, 
            "<6500 pixels by < 1500 pixels": 0, 
            "<6500 pixels by < 2000 pixels": 0, 
            "<6500 pixels by < 2500 pixels": 0, 
            "<6500 pixels by < 3000 pixels": 0, 
            "<6500 pixels by < 3500 pixels": 0, 
            "<6500 pixels by < 4000 pixels": 0, 
            "<6500 pixels by < 4500 pixels": 0, 
            "<6500 pixels by < 5000 pixels": 0, 
            "<6500 pixels by < 5500 pixels": 0, 
            "<65000 pixels by < 6000 pixels": 0, 
            "<65000 pixels by < 6500 pixels": 0, 
            "<65000 pixels by < 7000 pixels": 0, 
            "<65000 pixels by < 7500 pixels": 0,
            "<7000 pixels by < 500 pixels": 0, 
            "<7000 pixels by < 1000 pixels": 0, 
            "<7000 pixels by < 1500 pixels": 0, 
            "<7000 pixels by < 2000 pixels": 0, 
            "<7000 pixels by < 2500 pixels": 0, 
            "<7000 pixels by < 3000 pixels": 0, 
            "<7000 pixels by < 3500 pixels": 0, 
            "<7000 pixels by < 4000 pixels": 0, 
            "<7000 pixels by < 4500 pixels": 0, 
            "<7000 pixels by < 5000 pixels": 0, 
            "<7000 pixels by < 5500 pixels": 0, 
            "<7000 pixels by < 6000 pixels": 0, 
            "<7000 pixels by < 6500 pixels": 0, 
            "<7000 pixels by < 7000 pixels": 0, 
            "<7000 pixels by < 7500 pixels": 0,
            "<7500 pixels by < 500 pixels": 0, 
            "<7500 pixels by < 1000 pixels": 0, 
            "<7500 pixels by < 1500 pixels": 0, 
            "<7500 pixels by < 2000 pixels": 0, 
            "<7500 pixels by < 2500 pixels": 0, 
            "<7500 pixels by < 3000 pixels": 0, 
            "<7500 pixels by < 3500 pixels": 0, 
            "<7500 pixels by < 4000 pixels": 0, 
            "<7500 pixels by < 4500 pixels": 0, 
            "<7500 pixels by < 5000 pixels": 0, 
            "<7500 pixels by < 5500 pixels": 0, 
            "<7500 pixels by < 6000 pixels": 0, 
            "<7500 pixels by < 6500 pixels": 0, 
            "<7500 pixels by < 7000 pixels": 0, 
            "<7500 pixels by < 7500 pixels": 0,
            "<Uncategorized": 0,
}



def main(filepath):
    matches = []
    for root, dirnames, filenames in os.walk(filepath):
        for filename in fnmatch.filter(filenames, '*.png'):
            filepath = os.path.join(root, filename)
            matches.append(filepath)
            flood_filling(filepath)
    print(sizeDict)

def flood_filling(filepath):
    try:
        img = Image.open(filepath) 
        if (img.height <500 and img.width<500):
            current_count = sizeDict.get("<500 pixels by < 500 pixels") + 1
            sizeDict.update({"<500 pixels by < 500 pixels": current_count})
        elif (img.height <500 and img.width<1000):
            current_count = sizeDict.get("<500 pixels by < 1000 pixels") + 1
            sizeDict.update({"<500 pixels by < 1000 pixels": current_count})
        elif (img.height <500 and img.width<1500):
            current_count = sizeDict.get("<500 pixels by < 1500 pixels") + 1
            sizeDict.update({"<500 pixels by < 1500 pixels": current_count})
        elif (img.height <500 and img.width<2000):
            current_count = sizeDict.get("<500 pixels by < 2000 pixels") + 1
            sizeDict.update({"<500 pixels by < 2000 pixels": current_count})
        elif (img.height <500 and img.width<2500):
            current_count = sizeDict.get("<500 pixels by < 2500 pixels") + 1
            sizeDict.update({"<500 pixels by < 2500 pixels": current_count})
        elif (img.height <500 and img.width<3000):
            current_count = sizeDict.get("<500 pixels by < 3000 pixels") + 1
            sizeDict.update({"<500 pixels by < 3000 pixels": current_count})
        elif (img.height <500 and img.width<3500):
            current_count = sizeDict.get("<500 pixels by < 3500 pixels") + 1
            sizeDict.update({"<500 pixels by < 3500 pixels": current_count})
        elif (img.height <500 and img.width<4000):
            current_count = sizeDict.get("<500 pixels by < 4000 pixels") + 1
            sizeDict.update({"<500 pixels by < 4000 pixels": current_count})
        elif (img.height <500 and img.width<4500):
            current_count = sizeDict.get("<500 pixels by < 4500 pixels") + 1
            sizeDict.update({"<500 pixels by < 4500 pixels": current_count})            
        elif (img.height <500 and img.width<5000):
            current_count = sizeDict.get("<500 pixels by < 5000 pixels") + 1
            sizeDict.update({"<500 pixels by < 4500 pixels": current_count}) 
        elif (img.height <500 and img.width<5500):
            current_count = sizeDict.get("<500 pixels by < 5500 pixels") + 1
            sizeDict.update({"<500 pixels by < 5500 pixels": current_count}) 
        elif (img.height <500 and img.width<6000):
            current_count = sizeDict.get("<500 pixels by < 6000 pixels") + 1
            sizeDict.update({"<500 pixels by < 6000 pixels": current_count}) 
        elif (img.height <500 and img.width<6500):
            current_count = sizeDict.get("<500 pixels by < 6500 pixels") + 1
            sizeDict.update({"<500 pixels by < 5500 pixels": current_count}) 
        elif (img.height <500 and img.width<7000):
            current_count = sizeDict.get("<500 pixels by < 7000 pixels") + 1
            sizeDict.update({"<500 pixels by < 5500 pixels": current_count}) 
        elif (img.height <500 and img.width<7500):
            current_count = sizeDict.get("<500 pixels by < 7500 pixels") + 1
            sizeDict.update({"<500 pixels by < 7500 pixels": current_count}) 
        elif (img.height <1000 and img.width<500):
            current_count = sizeDict.get("<1000 pixels by < 500 pixels") + 1
            sizeDict.update({"<1000 pixels by < 500 pixels": current_count})
        elif (img.height <1000 and img.width<1000):
            current_count = sizeDict.get("<1000 pixels by < 1000 pixels") + 1
            sizeDict.update({"<1000 pixels by < 1000 pixels": current_count})
        elif (img.height <1000 and img.width<1500):
            current_count = sizeDict.get("<1000 pixels by < 1500 pixels") + 1
            sizeDict.update({"<1000 pixels by < 1500 pixels": current_count})
        elif (img.height <1000 and img.width<2000):
            current_count = sizeDict.get("<1000 pixels by < 2000 pixels") + 1
            sizeDict.update({"<1000 pixels by < 2000 pixels": current_count})
        elif (img.height <1000 and img.width<2500):
            current_count = sizeDict.get("<1000 pixels by < 2500 pixels") + 1
            sizeDict.update({"<1000 pixels by < 2500 pixels": current_count})
        elif (img.height <1000 and img.width<3000):
            current_count = sizeDict.get("<1000 pixels by < 3000 pixels") + 1
            sizeDict.update({"<1000 pixels by < 3000 pixels": current_count})
        elif (img.height <1000 and img.width<3500):
            current_count = sizeDict.get("<1000 pixels by < 3500 pixels") + 1
            sizeDict.update({"<1000 pixels by < 3500 pixels": current_count})
        elif (img.height <1000 and img.width<4000):
            current_count = sizeDict.get("<1000 pixels by < 4000 pixels") + 1
            sizeDict.update({"<1000 pixels by < 4000 pixels": current_count})
        elif (img.height <1000 and img.width<4500):
            current_count = sizeDict.get("<1000 pixels by < 4500 pixels") + 1
            sizeDict.update({"<1000 pixels by < 4500 pixels": current_count})            
        elif (img.height <1000 and img.width<5000):
            current_count = sizeDict.get("<1000 pixels by < 5000 pixels") + 1
            sizeDict.update({"<1000 pixels by < 4500 pixels": current_count}) 
        elif (img.height <1000 and img.width<5500):
            current_count = sizeDict.get("<1000 pixels by < 5500 pixels") + 1
            sizeDict.update({"<1000 pixels by < 5500 pixels": current_count}) 
        elif (img.height <1000 and img.width<6000):
            current_count = sizeDict.get("<1000 pixels by < 6000 pixels") + 1
            sizeDict.update({"<1000 pixels by < 6000 pixels": current_count}) 
        elif (img.height <1000 and img.width<6500):
            current_count = sizeDict.get("<1000 pixels by < 6500 pixels") + 1
            sizeDict.update({"<1000 pixels by < 5500 pixels": current_count}) 
        elif (img.height <1000 and img.width<7000):
            current_count = sizeDict.get("<1000 pixels by < 7000 pixels") + 1
            sizeDict.update({"<1000 pixels by < 5500 pixels": current_count}) 
        elif (img.height <1000 and img.width<7500):
            current_count = sizeDict.get("<1000 pixels by < 7500 pixels") + 1
            sizeDict.update({"<1000 pixels by < 7500 pixels": current_count}) 
        elif (img.height <1500 and img.width<500):
            current_count = sizeDict.get("<1500 pixels by < 500 pixels") + 1
            sizeDict.update({"<1500 pixels by < 500 pixels": current_count})
        elif (img.height <1500 and img.width<1000):
            current_count = sizeDict.get("<1500 pixels by < 1000 pixels") + 1
            sizeDict.update({"<1500 pixels by < 1000 pixels": current_count})
        elif (img.height <1500 and img.width<1500):
            current_count = sizeDict.get("<1500 pixels by < 1500 pixels") + 1
            sizeDict.update({"<1500 pixels by < 1500 pixels": current_count})
        elif (img.height <1500 and img.width<2000):
            current_count = sizeDict.get("<1500 pixels by < 2000 pixels") + 1
            sizeDict.update({"<1500 pixels by < 2000 pixels": current_count})
        elif (img.height <1500 and img.width<2500):
            current_count = sizeDict.get("<1500 pixels by < 2500 pixels") + 1
            sizeDict.update({"<1500 pixels by < 2500 pixels": current_count})
        elif (img.height <1500 and img.width<3000):
            current_count = sizeDict.get("<1500 pixels by < 3000 pixels") + 1
            sizeDict.update({"<1500 pixels by < 3000 pixels": current_count})
        elif (img.height <1500 and img.width<3500):
            current_count = sizeDict.get("<1500 pixels by < 3500 pixels") + 1
            sizeDict.update({"<1500 pixels by < 3500 pixels": current_count})
        elif (img.height <1500 and img.width<4000):
            current_count = sizeDict.get("<1500 pixels by < 4000 pixels") + 1
            sizeDict.update({"<1500 pixels by < 4000 pixels": current_count})
        elif (img.height <1500 and img.width<4500):
            current_count = sizeDict.get("<1500 pixels by < 4500 pixels") + 1
            sizeDict.update({"<1500 pixels by < 4500 pixels": current_count})            
        elif (img.height <1500 and img.width<5000):
            current_count = sizeDict.get("<1500 pixels by < 5000 pixels") + 1
            sizeDict.update({"<1500 pixels by < 4500 pixels": current_count}) 
        elif (img.height <1500 and img.width<5500):
            current_count = sizeDict.get("<1500 pixels by < 5500 pixels") + 1
            sizeDict.update({"<1500 pixels by < 5500 pixels": current_count}) 
        elif (img.height <1500 and img.width<6000):
            current_count = sizeDict.get("<1500 pixels by < 6000 pixels") + 1
            sizeDict.update({"<1500 pixels by < 6000 pixels": current_count}) 
        elif (img.height <1500 and img.width<6500):
            current_count = sizeDict.get("<1500 pixels by < 6500 pixels") + 1
            sizeDict.update({"<1500 pixels by < 5500 pixels": current_count}) 
        elif (img.height <1500 and img.width<7000):
            current_count = sizeDict.get("<1500 pixels by < 7000 pixels") + 1
            sizeDict.update({"<1500 pixels by < 5500 pixels": current_count}) 
        elif (img.height <1500 and img.width<7500):
            current_count = sizeDict.get("<1500 pixels by < 7500 pixels") + 1
            sizeDict.update({"<1500 pixels by < 7500 pixels": current_count}) 
        elif (img.height <2000 and img.width<500):
            current_count = sizeDict.get("<2000 pixels by < 500 pixels") + 1
            sizeDict.update({"<2000 pixels by < 500 pixels": current_count})
        elif (img.height <2000 and img.width<1000):
            current_count = sizeDict.get("<2000 pixels by < 1000 pixels") + 1
            sizeDict.update({"<2000 pixels by < 1000 pixels": current_count})
        elif (img.height <2000 and img.width<1500):
            current_count = sizeDict.get("<2000 pixels by < 1500 pixels") + 1
            sizeDict.update({"<2000 pixels by < 1500 pixels": current_count})
        elif (img.height <2000 and img.width<2000):
            current_count = sizeDict.get("<2000 pixels by < 2000 pixels") + 1
            sizeDict.update({"<2000 pixels by < 2000 pixels": current_count})
        elif (img.height <2000 and img.width<2500):
            current_count = sizeDict.get("<2000 pixels by < 2500 pixels") + 1
            sizeDict.update({"<2000 pixels by < 2500 pixels": current_count})
        elif (img.height <2000 and img.width<3000):
            current_count = sizeDict.get("<2000 pixels by < 3000 pixels") + 1
            sizeDict.update({"<2000 pixels by < 3000 pixels": current_count})
        elif (img.height <2000 and img.width<3500):
            current_count = sizeDict.get("<2000 pixels by < 3500 pixels") + 1
            sizeDict.update({"<2000 pixels by < 3500 pixels": current_count})
        elif (img.height <2000 and img.width<4000):
            current_count = sizeDict.get("<2000 pixels by < 4000 pixels") + 1
            sizeDict.update({"<2000 pixels by < 4000 pixels": current_count})
        elif (img.height <2000 and img.width<4500):
            current_count = sizeDict.get("<2000 pixels by < 4500 pixels") + 1
            sizeDict.update({"<2000 pixels by < 4500 pixels": current_count})            
        elif (img.height <2000 and img.width<5000):
            current_count = sizeDict.get("<2000 pixels by < 5000 pixels") + 1
            sizeDict.update({"<2000 pixels by < 4500 pixels": current_count}) 
        elif (img.height <2000 and img.width<5500):
            current_count = sizeDict.get("<2000 pixels by < 5500 pixels") + 1
            sizeDict.update({"<2000 pixels by < 5500 pixels": current_count}) 
        elif (img.height <2000 and img.width<6000):
            current_count = sizeDict.get("<2000 pixels by < 6000 pixels") + 1
            sizeDict.update({"<2000 pixels by < 6000 pixels": current_count}) 
        elif (img.height <2000 and img.width<6500):
            current_count = sizeDict.get("<2000 pixels by < 6500 pixels") + 1
            sizeDict.update({"<2000 pixels by < 5500 pixels": current_count}) 
        elif (img.height <2000 and img.width<7000):
            current_count = sizeDict.get("<2000 pixels by < 7000 pixels") + 1
            sizeDict.update({"<2000 pixels by < 5500 pixels": current_count}) 
        elif (img.height <2000 and img.width<7500):
            current_count = sizeDict.get("<2000 pixels by < 7500 pixels") + 1
            sizeDict.update({"<2000 pixels by < 7500 pixels": current_count}) 
        elif (img.height <2500 and img.width<500):
            current_count = sizeDict.get("<2500 pixels by < 500 pixels") + 1
            sizeDict.update({"<2500 pixels by < 500 pixels": current_count})
        elif (img.height <2500 and img.width<1000):
            current_count = sizeDict.get("<2500 pixels by < 1000 pixels") + 1
            sizeDict.update({"<2500 pixels by < 1000 pixels": current_count})
        elif (img.height <2500 and img.width<1500):
            current_count = sizeDict.get("<2500 pixels by < 1500 pixels") + 1
            sizeDict.update({"<2500 pixels by < 1500 pixels": current_count})
        elif (img.height <2500 and img.width<2000):
            current_count = sizeDict.get("<2500 pixels by < 2000 pixels") + 1
            sizeDict.update({"<2500 pixels by < 2000 pixels": current_count})
        elif (img.height <2500 and img.width<2500):
            current_count = sizeDict.get("<2500 pixels by < 2500 pixels") + 1
            sizeDict.update({"<2500 pixels by < 2500 pixels": current_count})
        elif (img.height <2500 and img.width<3000):
            current_count = sizeDict.get("<2500 pixels by < 3000 pixels") + 1
            sizeDict.update({"<2500 pixels by < 3000 pixels": current_count})
        elif (img.height <2500 and img.width<3500):
            current_count = sizeDict.get("<2500 pixels by < 3500 pixels") + 1
            sizeDict.update({"<2500 pixels by < 3500 pixels": current_count})
        elif (img.height <2500 and img.width<4000):
            current_count = sizeDict.get("<2500 pixels by < 4000 pixels") + 1
            sizeDict.update({"<2500 pixels by < 4000 pixels": current_count})
        elif (img.height <2500 and img.width<4500):
            current_count = sizeDict.get("<2500 pixels by < 4500 pixels") + 1
            sizeDict.update({"<2500 pixels by < 4500 pixels": current_count})            
        elif (img.height <2500 and img.width<5000):
            current_count = sizeDict.get("<2500 pixels by < 5000 pixels") + 1
            sizeDict.update({"<2500 pixels by < 4500 pixels": current_count}) 
        elif (img.height <2500 and img.width<5500):
            current_count = sizeDict.get("<2500 pixels by < 5500 pixels") + 1
            sizeDict.update({"<2500 pixels by < 5500 pixels": current_count}) 
        elif (img.height <2500 and img.width<6000):
            current_count = sizeDict.get("<2500 pixels by < 6000 pixels") + 1
            sizeDict.update({"<2500 pixels by < 6000 pixels": current_count}) 
        elif (img.height <2500 and img.width<6500):
            current_count = sizeDict.get("<2500 pixels by < 6500 pixels") + 1
            sizeDict.update({"<2500 pixels by < 5500 pixels": current_count}) 
        elif (img.height <2500 and img.width<7000):
            current_count = sizeDict.get("<2500 pixels by < 7000 pixels") + 1
            sizeDict.update({"<2500 pixels by < 5500 pixels": current_count}) 
        elif (img.height <2500 and img.width<7500):
            current_count = sizeDict.get("<2500 pixels by < 7500 pixels") + 1
            sizeDict.update({"<2500 pixels by < 7500 pixels": current_count})    
        elif (img.height <3000 and img.width<500):
            current_count = sizeDict.get("<3000 pixels by < 500 pixels") + 1
            sizeDict.update({"<3000 pixels by < 500 pixels": current_count})
        elif (img.height <3000 and img.width<1000):
            current_count = sizeDict.get("<3000 pixels by < 1000 pixels") + 1
            sizeDict.update({"<3000 pixels by < 1000 pixels": current_count})
        elif (img.height <3000 and img.width<1500):
            current_count = sizeDict.get("<3000 pixels by < 1500 pixels") + 1
            sizeDict.update({"<3000 pixels by < 1500 pixels": current_count})
        elif (img.height <3000 and img.width<2000):
            current_count = sizeDict.get("<3000 pixels by < 2000 pixels") + 1
            sizeDict.update({"<3000 pixels by < 2000 pixels": current_count})
        elif (img.height <3000 and img.width<2500):
            current_count = sizeDict.get("<3000 pixels by < 2500 pixels") + 1
            sizeDict.update({"<3000 pixels by < 2500 pixels": current_count})
        elif (img.height <3000 and img.width<3000):
            current_count = sizeDict.get("<3000 pixels by < 3000 pixels") + 1
            sizeDict.update({"<3000 pixels by < 3000 pixels": current_count})
        elif (img.height <3000 and img.width<3500):
            current_count = sizeDict.get("<3000 pixels by < 3500 pixels") + 1
            sizeDict.update({"<3000 pixels by < 3500 pixels": current_count})
        elif (img.height <3000 and img.width<4000):
            current_count = sizeDict.get("<3000 pixels by < 4000 pixels") + 1
            sizeDict.update({"<3000 pixels by < 4000 pixels": current_count})
        elif (img.height <3000 and img.width<4500):
            current_count = sizeDict.get("<3000 pixels by < 4500 pixels") + 1
            sizeDict.update({"<3000 pixels by < 4500 pixels": current_count})            
        elif (img.height <3000 and img.width<5000):
            current_count = sizeDict.get("<3000 pixels by < 5000 pixels") + 1
            sizeDict.update({"<3000 pixels by < 4500 pixels": current_count}) 
        elif (img.height <3000 and img.width<5500):
            current_count = sizeDict.get("<3000 pixels by < 5500 pixels") + 1
            sizeDict.update({"<3000 pixels by < 5500 pixels": current_count}) 
        elif (img.height <3000 and img.width<6000):
            current_count = sizeDict.get("<3000 pixels by < 6000 pixels") + 1
            sizeDict.update({"<3000 pixels by < 6000 pixels": current_count}) 
        elif (img.height <3000 and img.width<6500):
            current_count = sizeDict.get("<3000 pixels by < 6500 pixels") + 1
            sizeDict.update({"<3000 pixels by < 5500 pixels": current_count}) 
        elif (img.height <3000 and img.width<7000):
            current_count = sizeDict.get("<3000 pixels by < 7000 pixels") + 1
            sizeDict.update({"<3000 pixels by < 5500 pixels": current_count}) 
        elif (img.height <3000 and img.width<7500):
            current_count = sizeDict.get("<3000 pixels by < 7500 pixels") + 1
            sizeDict.update({"<3000 pixels by < 7500 pixels": current_count})   
        elif (img.height <3500 and img.width<500):
            current_count = sizeDict.get("<3500 pixels by < 500 pixels") + 1
            sizeDict.update({"<3500 pixels by < 500 pixels": current_count})
        elif (img.height <3500 and img.width<1000):
            current_count = sizeDict.get("<3500 pixels by < 1000 pixels") + 1
            sizeDict.update({"<3500 pixels by < 1000 pixels": current_count})
        elif (img.height <3500 and img.width<1500):
            current_count = sizeDict.get("<3500 pixels by < 1500 pixels") + 1
            sizeDict.update({"<3500 pixels by < 1500 pixels": current_count})
        elif (img.height <3500 and img.width<2000):
            current_count = sizeDict.get("<3500 pixels by < 2000 pixels") + 1
            sizeDict.update({"<3500 pixels by < 2000 pixels": current_count})
        elif (img.height <3500 and img.width<2500):
            current_count = sizeDict.get("<3500 pixels by < 2500 pixels") + 1
            sizeDict.update({"<3500 pixels by < 2500 pixels": current_count})
        elif (img.height <3500 and img.width<3000):
            current_count = sizeDict.get("<3500 pixels by < 3000 pixels") + 1
            sizeDict.update({"<3500 pixels by < 3000 pixels": current_count})
        elif (img.height <3500 and img.width<3500):
            current_count = sizeDict.get("<3500 pixels by < 3500 pixels") + 1
            sizeDict.update({"<3500 pixels by < 3500 pixels": current_count})
        elif (img.height <3500 and img.width<4000):
            current_count = sizeDict.get("<3500 pixels by < 4000 pixels") + 1
            sizeDict.update({"<3500 pixels by < 4000 pixels": current_count})
        elif (img.height <3500 and img.width<4500):
            current_count = sizeDict.get("<3500 pixels by < 4500 pixels") + 1
            sizeDict.update({"<3500 pixels by < 4500 pixels": current_count})            
        elif (img.height <3500 and img.width<5000):
            current_count = sizeDict.get("<3500 pixels by < 5000 pixels") + 1
            sizeDict.update({"<3500 pixels by < 4500 pixels": current_count}) 
        elif (img.height <3500 and img.width<5500):
            current_count = sizeDict.get("<3500 pixels by < 5500 pixels") + 1
            sizeDict.update({"<3500 pixels by < 5500 pixels": current_count}) 
        elif (img.height <3500 and img.width<6000):
            current_count = sizeDict.get("<3500 pixels by < 6000 pixels") + 1
            sizeDict.update({"<3500 pixels by < 6000 pixels": current_count}) 
        elif (img.height <3500 and img.width<6500):
            current_count = sizeDict.get("<3500 pixels by < 6500 pixels") + 1
            sizeDict.update({"<3500 pixels by < 5500 pixels": current_count}) 
        elif (img.height <3500 and img.width<7000):
            current_count = sizeDict.get("<3500 pixels by < 7000 pixels") + 1
            sizeDict.update({"<3500 pixels by < 5500 pixels": current_count}) 
        elif (img.height <3500 and img.width<7500):
            current_count = sizeDict.get("<3500 pixels by < 7500 pixels") + 1
            sizeDict.update({"<3500 pixels by < 7500 pixels": current_count}) 
        elif (img.height <4000 and img.width<500):
            current_count = sizeDict.get("<4000 pixels by < 500 pixels") + 1
            sizeDict.update({"<4000 pixels by < 500 pixels": current_count})
        elif (img.height <4000 and img.width<1000):
            current_count = sizeDict.get("<4000 pixels by < 1000 pixels") + 1
            sizeDict.update({"<4000 pixels by < 1000 pixels": current_count})
        elif (img.height <4000 and img.width<1500):
            current_count = sizeDict.get("<4000 pixels by < 1500 pixels") + 1
            sizeDict.update({"<4000 pixels by < 1500 pixels": current_count})
        elif (img.height <4000 and img.width<2000):
            current_count = sizeDict.get("<4000 pixels by < 2000 pixels") + 1
            sizeDict.update({"<4000 pixels by < 2000 pixels": current_count})
        elif (img.height <4000 and img.width<2500):
            current_count = sizeDict.get("<4000 pixels by < 2500 pixels") + 1
            sizeDict.update({"<4000 pixels by < 2500 pixels": current_count})
        elif (img.height <4000 and img.width<3000):
            current_count = sizeDict.get("<4000 pixels by < 3000 pixels") + 1
            sizeDict.update({"<4000 pixels by < 3000 pixels": current_count})
        elif (img.height <4000 and img.width<3500):
            current_count = sizeDict.get("<4000 pixels by < 3500 pixels") + 1
            sizeDict.update({"<4000 pixels by < 3500 pixels": current_count})
        elif (img.height <4000 and img.width<4000):
            current_count = sizeDict.get("<4000 pixels by < 4000 pixels") + 1
            sizeDict.update({"<4000 pixels by < 4000 pixels": current_count})
        elif (img.height <4000 and img.width<4500):
            current_count = sizeDict.get("<4000 pixels by < 4500 pixels") + 1
            sizeDict.update({"<4000 pixels by < 4500 pixels": current_count})            
        elif (img.height <4000 and img.width<5000):
            current_count = sizeDict.get("<4000 pixels by < 5000 pixels") + 1
            sizeDict.update({"<4000 pixels by < 4500 pixels": current_count}) 
        elif (img.height <4000 and img.width<5500):
            current_count = sizeDict.get("<4000 pixels by < 5500 pixels") + 1
            sizeDict.update({"<4000 pixels by < 5500 pixels": current_count}) 
        elif (img.height <4000 and img.width<6000):
            current_count = sizeDict.get("<4000 pixels by < 6000 pixels") + 1
            sizeDict.update({"<4000 pixels by < 6000 pixels": current_count}) 
        elif (img.height <4000 and img.width<6500):
            current_count = sizeDict.get("<4000 pixels by < 6500 pixels") + 1
            sizeDict.update({"<4000 pixels by < 5500 pixels": current_count}) 
        elif (img.height <4000 and img.width<7000):
            current_count = sizeDict.get("<4000 pixels by < 7000 pixels") + 1
            sizeDict.update({"<4000 pixels by < 5500 pixels": current_count}) 
        elif (img.height <4000 and img.width<7500):
            current_count = sizeDict.get("<4000 pixels by < 7500 pixels") + 1
            sizeDict.update({"<4000 pixels by < 7500 pixels": current_count})    
        elif (img.height <4500 and img.width<500):
            current_count = sizeDict.get("<4500 pixels by < 500 pixels") + 1
            sizeDict.update({"<4500 pixels by < 500 pixels": current_count})
        elif (img.height <4500 and img.width<1000):
            current_count = sizeDict.get("<4500 pixels by < 1000 pixels") + 1
            sizeDict.update({"<4500 pixels by < 1000 pixels": current_count})
        elif (img.height <4500 and img.width<1500):
            current_count = sizeDict.get("<4500 pixels by < 1500 pixels") + 1
            sizeDict.update({"<4500 pixels by < 1500 pixels": current_count})
        elif (img.height <4500 and img.width<2000):
            current_count = sizeDict.get("<4500 pixels by < 2000 pixels") + 1
            sizeDict.update({"<4500 pixels by < 2000 pixels": current_count})
        elif (img.height <4500 and img.width<2500):
            current_count = sizeDict.get("<4500 pixels by < 2500 pixels") + 1
            sizeDict.update({"<4500 pixels by < 2500 pixels": current_count})
        elif (img.height <4500 and img.width<3000):
            current_count = sizeDict.get("<4500 pixels by < 3000 pixels") + 1
            sizeDict.update({"<4500 pixels by < 3000 pixels": current_count})
        elif (img.height <4500 and img.width<3500):
            current_count = sizeDict.get("<4500 pixels by < 3500 pixels") + 1
            sizeDict.update({"<4500 pixels by < 3500 pixels": current_count})
        elif (img.height <4500 and img.width<4000):
            current_count = sizeDict.get("<4500 pixels by < 4000 pixels") + 1
            sizeDict.update({"<4500 pixels by < 4000 pixels": current_count})
        elif (img.height <4500 and img.width<4500):
            current_count = sizeDict.get("<4500 pixels by < 4500 pixels") + 1
            sizeDict.update({"<4500 pixels by < 4500 pixels": current_count})            
        elif (img.height <4500 and img.width<5000):
            current_count = sizeDict.get("<4500 pixels by < 5000 pixels") + 1
            sizeDict.update({"<4500 pixels by < 4500 pixels": current_count}) 
        elif (img.height <4500 and img.width<5500):
            current_count = sizeDict.get("<4500 pixels by < 5500 pixels") + 1
            sizeDict.update({"<4500 pixels by < 5500 pixels": current_count}) 
        elif (img.height <4500 and img.width<6000):
            current_count = sizeDict.get("<4500 pixels by < 6000 pixels") + 1
            sizeDict.update({"<4500 pixels by < 6000 pixels": current_count}) 
        elif (img.height <4500 and img.width<6500):
            current_count = sizeDict.get("<4500 pixels by < 6500 pixels") + 1
            sizeDict.update({"<4500 pixels by < 5500 pixels": current_count}) 
        elif (img.height <4500 and img.width<7000):
            current_count = sizeDict.get("<4500 pixels by < 7000 pixels") + 1
            sizeDict.update({"<4500 pixels by < 5500 pixels": current_count}) 
        elif (img.height <4500 and img.width<7500):
            current_count = sizeDict.get("<4500 pixels by < 7500 pixels") + 1
            sizeDict.update({"<4500 pixels by < 7500 pixels": current_count})       
        elif (img.height <5000 and img.width<500):
            current_count = sizeDict.get("<5000 pixels by < 500 pixels") + 1
            sizeDict.update({"<5000 pixels by < 500 pixels": current_count})
        elif (img.height <5000 and img.width<1000):
            current_count = sizeDict.get("<5000 pixels by < 1000 pixels") + 1
            sizeDict.update({"<5000 pixels by < 1000 pixels": current_count})
        elif (img.height <5000 and img.width<1500):
            current_count = sizeDict.get("<5000 pixels by < 1500 pixels") + 1
            sizeDict.update({"<5000 pixels by < 1500 pixels": current_count})
        elif (img.height <5000 and img.width<2000):
            current_count = sizeDict.get("<5000 pixels by < 2000 pixels") + 1
            sizeDict.update({"<5000 pixels by < 2000 pixels": current_count})
        elif (img.height <5000 and img.width<2500):
            current_count = sizeDict.get("<5000 pixels by < 2500 pixels") + 1
            sizeDict.update({"<5000 pixels by < 2500 pixels": current_count})
        elif (img.height <5000 and img.width<3000):
            current_count = sizeDict.get("<5000 pixels by < 3000 pixels") + 1
            sizeDict.update({"<5000 pixels by < 3000 pixels": current_count})
        elif (img.height <5000 and img.width<3500):
            current_count = sizeDict.get("<5000 pixels by < 3500 pixels") + 1
            sizeDict.update({"<5000 pixels by < 3500 pixels": current_count})
        elif (img.height <5000 and img.width<4000):
            current_count = sizeDict.get("<5000 pixels by < 4000 pixels") + 1
            sizeDict.update({"<5000 pixels by < 4000 pixels": current_count})
        elif (img.height <5000 and img.width<4500):
            current_count = sizeDict.get("<5000 pixels by < 4500 pixels") + 1
            sizeDict.update({"<5000 pixels by < 4500 pixels": current_count})            
        elif (img.height <5000 and img.width<5000):
            current_count = sizeDict.get("<5000 pixels by < 5000 pixels") + 1
            sizeDict.update({"<5000 pixels by < 4500 pixels": current_count}) 
        elif (img.height <5000 and img.width<5500):
            current_count = sizeDict.get("<5000 pixels by < 5500 pixels") + 1
            sizeDict.update({"<5000 pixels by < 5500 pixels": current_count}) 
        elif (img.height <5000 and img.width<6000):
            current_count = sizeDict.get("<5000 pixels by < 6000 pixels") + 1
            sizeDict.update({"<5000 pixels by < 6000 pixels": current_count}) 
        elif (img.height <5000 and img.width<6500):
            current_count = sizeDict.get("<5000 pixels by < 6500 pixels") + 1
            sizeDict.update({"<5000 pixels by < 5500 pixels": current_count}) 
        elif (img.height <5000 and img.width<7000):
            current_count = sizeDict.get("<5000 pixels by < 7000 pixels") + 1
            sizeDict.update({"<5000 pixels by < 5500 pixels": current_count}) 
        elif (img.height <5000 and img.width<7500):
            current_count = sizeDict.get("<5000 pixels by < 7500 pixels") + 1
            sizeDict.update({"<5000 pixels by < 7500 pixels": current_count})  
        elif (img.height <5500 and img.width<500):
            current_count = sizeDict.get("<5500 pixels by < 500 pixels") + 1
            sizeDict.update({"<5500 pixels by < 500 pixels": current_count})
        elif (img.height <5500 and img.width<1000):
            current_count = sizeDict.get("<5500 pixels by < 1000 pixels") + 1
            sizeDict.update({"<5500 pixels by < 1000 pixels": current_count})
        elif (img.height <5500 and img.width<1500):
            current_count = sizeDict.get("<5500 pixels by < 1500 pixels") + 1
            sizeDict.update({"<5500 pixels by < 1500 pixels": current_count})
        elif (img.height <5500 and img.width<2000):
            current_count = sizeDict.get("<5500 pixels by < 2000 pixels") + 1
            sizeDict.update({"<5500 pixels by < 2000 pixels": current_count})
        elif (img.height <5500 and img.width<2500):
            current_count = sizeDict.get("<5500 pixels by < 2500 pixels") + 1
            sizeDict.update({"<5500 pixels by < 2500 pixels": current_count})
        elif (img.height <5500 and img.width<3000):
            current_count = sizeDict.get("<5500 pixels by < 3000 pixels") + 1
            sizeDict.update({"<5500 pixels by < 3000 pixels": current_count})
        elif (img.height <5500 and img.width<3500):
            current_count = sizeDict.get("<5500 pixels by < 3500 pixels") + 1
            sizeDict.update({"<5500 pixels by < 3500 pixels": current_count})
        elif (img.height <5500 and img.width<4000):
            current_count = sizeDict.get("<5500 pixels by < 4000 pixels") + 1
            sizeDict.update({"<5500 pixels by < 4000 pixels": current_count})
        elif (img.height <5500 and img.width<4500):
            current_count = sizeDict.get("<5500 pixels by < 4500 pixels") + 1
            sizeDict.update({"<5500 pixels by < 4500 pixels": current_count})            
        elif (img.height <5500 and img.width<5000):
            current_count = sizeDict.get("<5500 pixels by < 5000 pixels") + 1
            sizeDict.update({"<5500 pixels by < 4500 pixels": current_count}) 
        elif (img.height <5500 and img.width<5500):
            current_count = sizeDict.get("<5500 pixels by < 5500 pixels") + 1
            sizeDict.update({"<5500 pixels by < 5500 pixels": current_count}) 
        elif (img.height <5500 and img.width<6000):
            current_count = sizeDict.get("<5500 pixels by < 6000 pixels") + 1
            sizeDict.update({"<5500 pixels by < 6000 pixels": current_count}) 
        elif (img.height <5500 and img.width<6500):
            current_count = sizeDict.get("<5500 pixels by < 6500 pixels") + 1
            sizeDict.update({"<5500 pixels by < 5500 pixels": current_count}) 
        elif (img.height <5500 and img.width<7000):
            current_count = sizeDict.get("<5500 pixels by < 7000 pixels") + 1
            sizeDict.update({"<5500 pixels by < 5500 pixels": current_count}) 
        elif (img.height <5500 and img.width<7500):
            current_count = sizeDict.get("<5500 pixels by < 7500 pixels") + 1
            sizeDict.update({"<5500 pixels by < 7500 pixels": current_count})       
        elif (img.height <6000 and img.width<500):
            current_count = sizeDict.get("<6000 pixels by < 500 pixels") + 1
            sizeDict.update({"<6000 pixels by < 500 pixels": current_count})
        elif (img.height <6000 and img.width<1000):
            current_count = sizeDict.get("<6000 pixels by < 1000 pixels") + 1
            sizeDict.update({"<6000 pixels by < 1000 pixels": current_count})
        elif (img.height <6000 and img.width<1500):
            current_count = sizeDict.get("<6000 pixels by < 1500 pixels") + 1
            sizeDict.update({"<6000 pixels by < 1500 pixels": current_count})
        elif (img.height <6000 and img.width<2000):
            current_count = sizeDict.get("<6000 pixels by < 2000 pixels") + 1
            sizeDict.update({"<6000 pixels by < 2000 pixels": current_count})
        elif (img.height <6000 and img.width<2500):
            current_count = sizeDict.get("<6000 pixels by < 2500 pixels") + 1
            sizeDict.update({"<6000 pixels by < 2500 pixels": current_count})
        elif (img.height <6000 and img.width<3000):
            current_count = sizeDict.get("<6000 pixels by < 3000 pixels") + 1
            sizeDict.update({"<6000 pixels by < 3000 pixels": current_count})
        elif (img.height <6000 and img.width<3500):
            current_count = sizeDict.get("<6000 pixels by < 3500 pixels") + 1
            sizeDict.update({"<6000 pixels by < 3500 pixels": current_count})
        elif (img.height <6000 and img.width<4000):
            current_count = sizeDict.get("<6000 pixels by < 4000 pixels") + 1
            sizeDict.update({"<6000 pixels by < 4000 pixels": current_count})
        elif (img.height <6000 and img.width<4500):
            current_count = sizeDict.get("<6000 pixels by < 4500 pixels") + 1
            sizeDict.update({"<6000 pixels by < 4500 pixels": current_count})            
        elif (img.height <6000 and img.width<5000):
            current_count = sizeDict.get("<6000 pixels by < 5000 pixels") + 1
            sizeDict.update({"<6000 pixels by < 4500 pixels": current_count}) 
        elif (img.height <6000 and img.width<5500):
            current_count = sizeDict.get("<6000 pixels by < 5500 pixels") + 1
            sizeDict.update({"<6000 pixels by < 5500 pixels": current_count}) 
        elif (img.height <6000 and img.width<6000):
            current_count = sizeDict.get("<6000 pixels by < 6000 pixels") + 1
            sizeDict.update({"<6000 pixels by < 6000 pixels": current_count}) 
        elif (img.height <6000 and img.width<6500):
            current_count = sizeDict.get("<6000 pixels by < 6500 pixels") + 1
            sizeDict.update({"<6000 pixels by < 5500 pixels": current_count}) 
        elif (img.height <6000 and img.width<7000):
            current_count = sizeDict.get("<6000 pixels by < 7000 pixels") + 1
            sizeDict.update({"<6000 pixels by < 5500 pixels": current_count}) 
        elif (img.height <6000 and img.width<7500):
            current_count = sizeDict.get("<6000 pixels by < 7500 pixels") + 1
            sizeDict.update({"<6000 pixels by < 7500 pixels": current_count})   
        elif (img.height <6500 and img.width<500):
            current_count = sizeDict.get("<6500 pixels by < 500 pixels") + 1
            sizeDict.update({"<6500 pixels by < 500 pixels": current_count})
        elif (img.height <6500 and img.width<1000):
            current_count = sizeDict.get("<6500 pixels by < 1000 pixels") + 1
            sizeDict.update({"<6500 pixels by < 1000 pixels": current_count})
        elif (img.height <6500 and img.width<1500):
            current_count = sizeDict.get("<6500 pixels by < 1500 pixels") + 1
            sizeDict.update({"<6500 pixels by < 1500 pixels": current_count})
        elif (img.height <6500 and img.width<2000):
            current_count = sizeDict.get("<6500 pixels by < 2000 pixels") + 1
            sizeDict.update({"<6500 pixels by < 2000 pixels": current_count})
        elif (img.height <6500 and img.width<2500):
            current_count = sizeDict.get("<6500 pixels by < 2500 pixels") + 1
            sizeDict.update({"<6500 pixels by < 2500 pixels": current_count})
        elif (img.height <6500 and img.width<3000):
            current_count = sizeDict.get("<6500 pixels by < 3000 pixels") + 1
            sizeDict.update({"<6500 pixels by < 3000 pixels": current_count})
        elif (img.height <6500 and img.width<3500):
            current_count = sizeDict.get("<6500 pixels by < 3500 pixels") + 1
            sizeDict.update({"<6500 pixels by < 3500 pixels": current_count})
        elif (img.height <6500 and img.width<4000):
            current_count = sizeDict.get("<6500 pixels by < 4000 pixels") + 1
            sizeDict.update({"<6500 pixels by < 4000 pixels": current_count})
        elif (img.height <6500 and img.width<4500):
            current_count = sizeDict.get("<6500 pixels by < 4500 pixels") + 1
            sizeDict.update({"<6500 pixels by < 4500 pixels": current_count})            
        elif (img.height <6500 and img.width<5000):
            current_count = sizeDict.get("<6500 pixels by < 5000 pixels") + 1
            sizeDict.update({"<6500 pixels by < 4500 pixels": current_count}) 
        elif (img.height <6500 and img.width<5500):
            current_count = sizeDict.get("<6500 pixels by < 5500 pixels") + 1
            sizeDict.update({"<6500 pixels by < 5500 pixels": current_count}) 
        elif (img.height <6500 and img.width<6000):
            current_count = sizeDict.get("<6500 pixels by < 6000 pixels") + 1
            sizeDict.update({"<6500 pixels by < 6000 pixels": current_count}) 
        elif (img.height <6500 and img.width<6500):
            current_count = sizeDict.get("<6500 pixels by < 6500 pixels") + 1
            sizeDict.update({"<6500 pixels by < 5500 pixels": current_count}) 
        elif (img.height <6500 and img.width<7000):
            current_count = sizeDict.get("<6500 pixels by < 7000 pixels") + 1
            sizeDict.update({"<6500 pixels by < 5500 pixels": current_count}) 
        elif (img.height <6500 and img.width<7500):
            current_count = sizeDict.get("<6500 pixels by < 7500 pixels") + 1
            sizeDict.update({"<6500 pixels by < 7500 pixels": current_count})   
        elif (img.height <7000 and img.width<500):
            current_count = sizeDict.get("<7000 pixels by < 500 pixels") + 1
            sizeDict.update({"<7000 pixels by < 500 pixels": current_count})
        elif (img.height <7000 and img.width<1000):
            current_count = sizeDict.get("<7000 pixels by < 1000 pixels") + 1
            sizeDict.update({"<7000 pixels by < 1000 pixels": current_count})
        elif (img.height <7000 and img.width<1500):
            current_count = sizeDict.get("<7000 pixels by < 1500 pixels") + 1
            sizeDict.update({"<7000 pixels by < 1500 pixels": current_count})
        elif (img.height <7000 and img.width<2000):
            current_count = sizeDict.get("<7000 pixels by < 2000 pixels") + 1
            sizeDict.update({"<7000 pixels by < 2000 pixels": current_count})
        elif (img.height <7000 and img.width<2500):
            current_count = sizeDict.get("<7000 pixels by < 2500 pixels") + 1
            sizeDict.update({"<7000 pixels by < 2500 pixels": current_count})
        elif (img.height <7000 and img.width<3000):
            current_count = sizeDict.get("<7000 pixels by < 3000 pixels") + 1
            sizeDict.update({"<7000 pixels by < 3000 pixels": current_count})
        elif (img.height <7000 and img.width<3500):
            current_count = sizeDict.get("<7000 pixels by < 3500 pixels") + 1
            sizeDict.update({"<7000 pixels by < 3500 pixels": current_count})
        elif (img.height <7000 and img.width<4000):
            current_count = sizeDict.get("<7000 pixels by < 4000 pixels") + 1
            sizeDict.update({"<7000 pixels by < 4000 pixels": current_count})
        elif (img.height <7000 and img.width<4500):
            current_count = sizeDict.get("<7000 pixels by < 4500 pixels") + 1
            sizeDict.update({"<7000 pixels by < 4500 pixels": current_count})            
        elif (img.height <7000 and img.width<5000):
            current_count = sizeDict.get("<7000 pixels by < 5000 pixels") + 1
            sizeDict.update({"<7000 pixels by < 4500 pixels": current_count}) 
        elif (img.height <7000 and img.width<5500):
            current_count = sizeDict.get("<7000 pixels by < 5500 pixels") + 1
            sizeDict.update({"<7000 pixels by < 5500 pixels": current_count}) 
        elif (img.height <7000 and img.width<6000):
            current_count = sizeDict.get("<7000 pixels by < 6000 pixels") + 1
            sizeDict.update({"<7000 pixels by < 6000 pixels": current_count}) 
        elif (img.height <7000 and img.width<6500):
            current_count = sizeDict.get("<7000 pixels by < 6500 pixels") + 1
            sizeDict.update({"<7000 pixels by < 5500 pixels": current_count}) 
        elif (img.height <7000 and img.width<7000):
            current_count = sizeDict.get("<7000 pixels by < 7000 pixels") + 1
            sizeDict.update({"<7000 pixels by < 5500 pixels": current_count}) 
        elif (img.height <7000 and img.width<7500):
            current_count = sizeDict.get("<7000 pixels by < 7500 pixels") + 1
            sizeDict.update({"<7000 pixels by < 7500 pixels": current_count})      
        elif (img.height <7500 and img.width<500):
            current_count = sizeDict.get("<7500 pixels by < 500 pixels") + 1
            sizeDict.update({"<7500 pixels by < 500 pixels": current_count})
        elif (img.height <7500 and img.width<1000):
            current_count = sizeDict.get("<7500 pixels by < 1000 pixels") + 1
            sizeDict.update({"<7500 pixels by < 1000 pixels": current_count})
        elif (img.height <7500 and img.width<1500):
            current_count = sizeDict.get("<7500 pixels by < 1500 pixels") + 1
            sizeDict.update({"<7500 pixels by < 1500 pixels": current_count})
        elif (img.height <7500 and img.width<2000):
            current_count = sizeDict.get("<7500 pixels by < 2000 pixels") + 1
            sizeDict.update({"<7500 pixels by < 2000 pixels": current_count})
        elif (img.height <7500 and img.width<2500):
            current_count = sizeDict.get("<7500 pixels by < 2500 pixels") + 1
            sizeDict.update({"<7500 pixels by < 2500 pixels": current_count})
        elif (img.height <7500 and img.width<3000):
            current_count = sizeDict.get("<7500 pixels by < 3000 pixels") + 1
            sizeDict.update({"<7500 pixels by < 3000 pixels": current_count})
        elif (img.height <7500 and img.width<3500):
            current_count = sizeDict.get("<7500 pixels by < 3500 pixels") + 1
            sizeDict.update({"<7500 pixels by < 3500 pixels": current_count})
        elif (img.height <7500 and img.width<4000):
            current_count = sizeDict.get("<7500 pixels by < 4000 pixels") + 1
            sizeDict.update({"<7500 pixels by < 4000 pixels": current_count})
        elif (img.height <7500 and img.width<4500):
            current_count = sizeDict.get("<7500 pixels by < 4500 pixels") + 1
            sizeDict.update({"<7500 pixels by < 4500 pixels": current_count})            
        elif (img.height <7500 and img.width<5000):
            current_count = sizeDict.get("<7500 pixels by < 5000 pixels") + 1
            sizeDict.update({"<7500 pixels by < 4500 pixels": current_count}) 
        elif (img.height <7500 and img.width<5500):
            current_count = sizeDict.get("<7500 pixels by < 5500 pixels") + 1
            sizeDict.update({"<7500 pixels by < 5500 pixels": current_count}) 
        elif (img.height <7500 and img.width<6000):
            current_count = sizeDict.get("<7500 pixels by < 6000 pixels") + 1
            sizeDict.update({"<7500 pixels by < 6000 pixels": current_count}) 
        elif (img.height <7500 and img.width<6500):
            current_count = sizeDict.get("<7500 pixels by < 6500 pixels") + 1
            sizeDict.update({"<7500 pixels by < 5500 pixels": current_count}) 
        elif (img.height <7500 and img.width<7000):
            current_count = sizeDict.get("<7500 pixels by < 7000 pixels") + 1
            sizeDict.update({"<7500 pixels by < 5500 pixels": current_count}) 
        elif (img.height <7500 and img.width<7500):
            current_count = sizeDict.get("<7500 pixels by < 7500 pixels") + 1
            sizeDict.update({"<7500 pixels by < 7500 pixels": current_count})     
        else:   
            current_count = sizeDict.get("Uncategorized") + 1
            sizeDict.update({"Uncategorized": current_count})                     
    except:
        err = sys.exc_info()[0]
        print(str(err))


# filepath: D:\dataset\manifest-ZkhPvrLo5216730872708713142
main('C:/Users/andre/OneDrive/Desktop/ccmlo_enhanced/cc/test')