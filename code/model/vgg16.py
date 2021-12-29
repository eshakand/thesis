import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Dense, MaxPool2D , Flatten
from tensorflow.keras import regularizers
from tensorflow.keras.constraints import unit_norm
from tensorflow.keras.optimizers import Nadam, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.constraints import UnitNorm
from PIL import ImageFile
import matplotlib.pyplot as plt
from keras.models import model_from_json 
import pickle 
import os
ImageFile.LOAD_TRUNCATED_IMAGES = True
from numba import jit

@jit
def main():
	central_storage_strategy = tf.distribute.MirroredStrategy()

	trdata = ImageDataGenerator()
	traindataimages = trdata.flow_from_directory(directory=os.getenv("training_directory"), target_size=(5500, 5500), batch_size=2,color_mode='grayscale')

	testdata = ImageDataGenerator(fill_mode = 'constant')#:rotation_range=15, horizontal_flip=True, vertical_flip=True)#rotation_range=180)
	testdataimages = testdata.flow_from_directory(directory=os.getenv("test_directory"), target_size=(5500, 5500), batch_size=2,color_mode='grayscale')

	model_0=Sequential()

	model_0.add(Conv2D(filters=4, kernel_size=(3, 3), padding='same', strides=(2,2), activation='relu',input_shape=(5500, 5500,1)))
	model_0.add(Conv2D(filters=4, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
	model_0.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
	model_0.add(BatchNormalization())

	model_0.add(Conv2D(filters=8, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
	model_0.add(Conv2D(filters=8, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
	model_0.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
	model_0.add(BatchNormalization())

	model_0.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
	model_0.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
	model_0.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
	model_0.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
	model_0.add(BatchNormalization())

	model_0.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
	model_0.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
	model_0.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
	model_0.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
	model_0.add(BatchNormalization())

	model_0.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
	model_0.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
	model_0.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
	model_0.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
	model_0.add(BatchNormalization())    
	model_0.add(Flatten())

	model_0.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
	model_0.add(Dropout(0.6))
	model_0.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
	model_0.add(Dropout(0.4))
	model_0.add(Dense(2, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001)))

	es = EarlyStopping(monitor='loss', mode='min', patience=10)
	model_0.summary()

	opt = Adam(lr = 1e-4)#Nadam()

	checkpoint = ModelCheckpoint(os.getenv("model_path"), monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', period=1)
	# Compile the model
	model_0.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	model_results = model_0.fit(traindataimages, batch_size=2, epochs=20, validation_data=testdataimages, validation_batch_size=2, callbacks=[checkpoint, es])
		

main()