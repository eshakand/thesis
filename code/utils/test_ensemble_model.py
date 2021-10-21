import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import numpy as np



model1 = tf.keras.models.load_model('./ccmodel1.hdf5')
model2 = tf.keras.models.load_model('./resnet50_v2.hdf5')
model3 = tf.keras.models.load_model('./ccmodel3.hdf5')
model4 = tf.keras.models.load_model('./model_v2.hdf5')
model6 = tf.keras.models.load_model('./ccmodel6.hdf5')
model7 = tf.keras.models.load_model('./ccmodel7.hdf5')
model9 = tf.keras.models.load_model('./ccmodel9.hdf5')
model15 = tf.keras.models.load_model('./ccmodel15.hdf5')



trdata = ImageDataGenerator()
test_images = trdata.flow_from_directory(directory="../first training/cnn_dataset_test",batch_size=1, target_size=(2900,4000), color_mode='grayscale')
test_images2 = trdata.flow_from_directory(directory="../first training/cnn_dataset_test",batch_size=1, target_size=(7125,7125), color_mode='grayscale')

print(len(test_images.filenames))
incorrectCount=0
for i in range(len(test_images.filenames)): 
        model1_prediction = model1.predict_classes(test_images[i][0])
        model3_prediction = model3.predict_classes(test_images[i][0])
        model4_prediction = model4.predict_classes(test_images2[i][0])
        model7_prediction = model7.predict_classes(test_images[i][0])
        model9_prediction = model9.predict_classes(test_images[i][0])
        model15_prediction = model15.predict_classes(test_images[i][0])
        

        benign_prediction_count=0
        predictions = [model1_prediction, model4_prediction, model3_prediction,  model7_prediction , model9_prediction ]#, model10_prediction]
        
        image_label = "benign"
        if (test_images[i][1][0][0]==1 and test_images[i][1][0][1]==0):
            image_label = "malignant"

        for prediction in predictions: 
            if prediction[0]==1:
                benign_prediction_count=benign_prediction_count+1

        textPrediction=""
        if (benign_prediction_count < 2):
            textPrediction = "malignant"
        else: 
            textPrediction = "benign"

        if (image_label != textPrediction):
            incorrectCount=incorrectCount+1
        print("Label is: %s. Prediction is: %s " % (image_label, textPrediction))

#test_loss, test_acc = full_resolution_model.evaluate(test_images,  batch_size=5, verbose=2)
print(incorrectCount)