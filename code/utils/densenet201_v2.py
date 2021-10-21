import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, MaxPool2D , Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from PIL import ImageFile
from tensorflow.keras.applications.densenet import *


central_storage_strategy = tf.distribute.MirroredStrategy()

trdata = ImageDataGenerator()#:rotation_range=15, horizontal_flip=True, vertical_flip=True)#rotation_range=180)
traindata = trdata.flow_from_directory(directory="./cc/training/",  target_size=(1920,1080), batch_size=1,color_mode='grayscale')


with central_storage_strategy.scope():
	model_0 = DenseNet201(input_shape=(1920,1080, 1), weights=None, pooling='max', classes=2)

	model_0.summary()

	checkpoint = ModelCheckpoint("densenet201_v2_local.hdf5", monitor='accuracy', verbose=1, save_best_only=True, mode='auto', period=1)
	# Compile the model
	model_0.compile(optimizer='Nadam', loss='binary_crossentropy', metrics=['accuracy'])

	model_0.fit(traindata, batch_size=1, steps_per_epoch=200, epochs=250, callbacks=[checkpoint])
#
#model_0.fit_gf.keras.metricsenerator(generator=traindata,
#                    use_multiprocessing=True,
#                    workers=6,max_queue_size=1)

trdata = ImageDataGenerator()
#
test_images = trdata.flow_from_directory(directory="/home-1/aeshak1@jhu.edu/scratch/full-resolution/test",batch_size=1, target_size=(2900,4000), color_mode='rgb')
test_loss, test_acc = model_0.evaluate(test_images,  batch_size=1, verbose=2)
model_0.save('model22.h5') 
