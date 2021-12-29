import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import numpy as np
import fnmatch
from keras.preprocessing import image

def main():
    filepath = os.getenv("dataset_path")
    matches = []
    incorrect_count = 0
    model1 = tf.keras.models.load_model(os.getenv("first_model_path"))
    model2 = tf.keras.models.load_model(os.getenv("second_model_path"))
    model3 = tf.keras.models.load_model(os.getenv("third_model_path"))
    model4 = tf.keras.models.load_model(os.getenv("fourth_model_path"))
    model5 = tf.keras.models.load_model(os.getenv("fifth_model_path"))
    model6 = tf.keras.models.load_model(os.getenv("sixth_model_path"))
    model7 = tf.keras.models.load_model(os.getenv("seventh_model_path"))
    model8 = tf.keras.models.load_model(os.getenv("eighth_model_path"))

    for root, dirnames, filenames in os.walk(filepath):
        for filename in fnmatch.filter(filenames, '*.png'):
            filepath = os.path.join(root, filename)
            matches.append(filepath)
            img = image.load_img(filepath, target_size=(5500, 5500), color_mode = 'grayscale')
            y = image.img_to_array(img)
            y = np.expand_dims(y, axis=0)
            model1_prediction = np.argmax(model1.predict(y), axis=0)
            model2_prediction = np.argmax(model2.predict(y), axis=0)
            model3_prediction = np.argmax(model3.predict(y), axis=0)
            model4_prediction = np.argmax(model4.predict(y), axis=0)
            model5_prediction = np.argmax(model5.predict(y), axis=0)
            model6_prediction = np.argmax(model6.predict(y), axis=0)
            model7_prediction = np.argmax(model7.predict(y), axis=0)
            model8_prediction = np.argmax(model8.predict(y), axis=0)
            benign_prediction_count=0
            malignant_prediction_count = 0

            predictions = [model1_prediction, model2_prediction, model3_prediction, model4_prediction, model5_prediction, model6_prediction, model7_prediction , model8_prediction] #, model9_prediction,  model10_prediction]#, model10_prediction]
        
            image_label = "benign"
            if ('benign' not in filepath):
                image_label = "malignant"

            for prediction in predictions:
                if prediction==0:
                    benign_prediction_count = benign_prediction_count + 1
                else:
                     malignant_prediction_count = malignant_prediction_count + 1

            text_prediction = ""
            if malignant_prediction_count > benign_prediction_count:
                text_prediction = "malignant"
            else:
                text_prediction = "benign"
            if (image_label != text_prediction):
                incorrect_count=incorrect_count+1
                print(filepath)
            print("Label is: %s. First model: %s. Second model: %s. Third model: %s. Fourth model: %s.  Fifth model: %s.  Sixth model: %s. Seventh model: %s.  eighth model: %s. Prediction is %s " % (image_label, model1_prediction, model2_prediction, model3_prediction, model4_prediction, model5_prediction, model6_prediction, model7_prediction, model8_prediction, text_prediction))
    print(incorrect_count)

main()