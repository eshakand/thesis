import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Dense, MaxPool2D , Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from PIL import ImageFile
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True


central_storage_strategy = tf.distribute.MirroredStrategy()

trdata = ImageDataGenerator(fill_mode = 'constant', shear_range=10, zoom_range=0.2)#:rotation_range=15, horizontal_flip=True, vertical_flip=True)#rotation_range=180)
traindataimages = trdata.flow_from_directory(directory="/home-1/aeshak1@jhu.edu/scratch/cc-mlo/cc/training", target_size=(5500, 5500), batch_size=4,color_mode='grayscale')

testdata = ImageDataGenerator(fill_mode = 'constant')#:rotation_range=15, horizontal_flip=True, vertical_flip=True)#rotation_range=180)
testdataimages = trdata.flow_from_directory(directory="/home-1/aeshak1@jhu.edu/scratch/cc-mlo/cc/test", target_size=(5500, 5500), batch_size=4,color_mode='grayscale')


with central_storage_strategy.scope():
        #model_0 = Sequential()

        model_0=Sequential()

        model_0.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', strides=(3,3), activation='relu', input_shape=(5500, 5500,1)))
        model_0.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(2,2), activation='relu'))
        model_0.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
        model_0.add(BatchNormalization());

        model_0.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(2,2), activation='relu'))
        model_0.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
        model_0.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
        model_0.add(BatchNormalization());

        model_0.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
        model_0.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
        model_0.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
        model_0.add(BatchNormalization());

        model_0.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
        model_0.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
        model_0.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
        model_0.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
        model_0.add(BatchNormalization());

        model_0.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
        model_0.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
        model_0.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
        model_0.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
        model_0.add(BatchNormalization());

        model_0.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
        model_0.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
        model_0.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
        model_0.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
        model_0.add(BatchNormalization());


        model_0.add(Flatten())

        model_0.add(Dense(4096, activation='relu'))
        model_0.add(Dropout(0.75))
        model_0.add(Dense(4096, activation='relu'))
        model_0.add(Dropout(0.75))
        model_0.add(Dense(2, activation='sigmoid'))

        es = EarlyStopping(monitor='loss', mode='min', patience=20)

        model_0.summary()


        checkpoint = ModelCheckpoint("ccmodel_v5.hdf5", monitor='loss', verbose=1, save_best_only=True, mode='auto', period=1)
