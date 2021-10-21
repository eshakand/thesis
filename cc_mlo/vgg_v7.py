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

trdata = ImageDataGenerator(fill_mode = 'constant')#:rotation_range=15, horizontal_flip=True, vertical_flip=True)#rotation_range=180)
traindataimages = trdata.flow_from_directory(directory="/home-1/aeshak1@jhu.edu/scratch/ccmlo2/cc/Training", target_size=(5500, 5500), batch_size=2,color_mode='grayscale')

testdata = ImageDataGenerator(fill_mode = 'constant')#:rotation_range=15, horizontal_flip=True, vertical_flip=True)#rotation_range=180)
testdataimages = trdata.flow_from_directory(directory="/home-1/aeshak1@jhu.edu/scratch/ccmlo2/cc/Test", target_size=(5500, 5500), batch_size=2,color_mode='grayscale')


with central_storage_strategy.scope():
        #model_0 = Sequential()

        model_0=Sequential()

        model_0.add(Conv2D(filters=2, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu', input_shape=(5500,5500,1)))
        model_0.add(Conv2D(filters=2, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
        model_0.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
        model_0.add(BatchNormalization());


        model_0.add(Conv2D(filters=4, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
        model_0.add(Conv2D(filters=4, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
        model_0.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
        model_0.add(BatchNormalization());

        model_0.add(Conv2D(filters=8, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
        model_0.add(Conv2D(filters=8, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
        model_0.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
        model_0.add(BatchNormalization());

        model_0.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
        model_0.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
        model_0.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
        model_0.add(BatchNormalization());

        model_0.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
        model_0.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
        model_0.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
        model_0.add(BatchNormalization());

        model_0.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
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
        model_0.add(Dense(4096, activation='relu'))
        model_0.add(Dense(2, activation='sigmoid'))

        es = EarlyStopping(monitor='loss', mode='min', patience=20)

        model_0.summary()


        checkpoint = ModelCheckpoint("ccmodel_v7.hdf5", monitor='loss', verbose=1, save_best_only=True, mode='auto', period=1)
        # Compile the model
        model_0.compile(optimizer='Nadam', loss='binary_crossentropy', metrics=['accuracy'])

        model_results = model_0.fit(traindataimages, batch_size=2, epochs=125, validation_data=testdataimages, validation_batch_size=2, validation_freq=5, callbacks=[checkpoint])


        plt.plot(model_results.history['accuracy'])
        plt.plot(model_results.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='upper left')
        plt.savefig('ccmodel-accuracy-v7.png')

        plt.plot(model_results.history['loss'])
        plt.plot(model_results.history['val_loss'])
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.legend(['Training Loss', 'Validation Loss'], loc='upper left')
        plt.savefig('ccmodel-loss-v7.png')
