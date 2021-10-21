import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, MaxPool2D , Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

central_storage_strategy = tf.distribute.MirroredStrategy()

trdata = ImageDataGenerator()#:rotation_range=15, horizontal_flip=True, vertical_flip=True)#rotation_range=180)
traindata = trdata.flow_from_directory(directory="/home-1/aeshak1@jhu.edu/scratch/roi_training/training", target_size=(400,400), batch_size=1,color_mode='grayscale')


with central_storage_strategy.scope():
        model_0 = Sequential()


        model_0.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu', input_shape=(400,400,1)))
        model_0.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
        model_0.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))


        model_0.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
        model_0.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
        model_0.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

        model_0.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu'))
        model_0.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu'))
        model_0.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu'))
        model_0.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

        model_0.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu'))
        model_0.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu'))
        model_0.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu'))
        model_0.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

        model_0.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
        model_0.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
        model_0.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
        model_0.add(MaxPooling2D((2,2)))


        model_0.add(Flatten())

        model_0.add(Dense(4096, activation='relu'))
        model_0.add(Dense(4096, activation='relu'))
        model_0.add(Dense(2, activation='sigmoid'))

        es = EarlyStopping(monitor='loss', mode='min', patience=20)

        model_0.summary()


        checkpoint = ModelCheckpoint("vgg16.hdf5", monitor='accuracy', verbose=1, save_best_only=True, mode='auto', period=1)
        # Compile the model
        model_0.compile(optimizer='Nadam', loss='binary_crossentropy', metrics=['accuracy'])

        model_0.fit(traindata, batch_size=1, epochs=250, callbacks=[checkpoint])
#
#model_0.fit_gf.keras.metricsenerator(generator=traindata,
#                    use_multiprocessing=True,
#                    workers=6,max_queue_size=1)

trdata = ImageDataGenerator()
#
test_images = trdata.flow_from_directory(directory="/home-1/aeshak1@jhu.edu/scratch/full-resolution/test",batch_size=1, target_size=(2900,4000), color_mode='rgb')
test_loss, test_acc = model_0.evaluate(test_images,  batch_size=1, verbose=2)
model_0.save('model22.h5')
                                           