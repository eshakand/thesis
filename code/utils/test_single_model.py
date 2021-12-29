import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import *
import os
import cv2
import numpy as np



model1 = tf.keras.models.load_model(os.getenv("model_path"))

trdata = ImageDataGenerator()

test_images = trdata.flow_from_directory(directory=os.getenv("dataset_path"),batch_size=2, target_size=(5500,5500), color_mode='grayscale')


test_loss, test_acc = model1.evaluate(test_images,  batch_size=2, verbose=2)
