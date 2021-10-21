import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import *
import os
import cv2
import numpy as np



model1 = tf.keras.models.load_model('./model_v4.hdf5')

trdata = ImageDataGenerator()

test_images = trdata.flow_from_directory(directory="C:/Users/andre/OneDrive/Desktop/ccmlo_enhanced/cc/test",batch_size=1, target_size=(7125,7125), color_mode='grayscale')


test_loss, test_acc = model1.evaluate(test_images,  batch_size=5, verbose=2)
