import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2DTranspose, Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Dense, MaxPool2D , Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

central_storage_strategy = tf.distribute.MirroredStrategy()

trdata = ImageDataGenerator()#:rotation_range=15, horizontal_flip=True, vertical_flip=True)#rotation_range=180)
traindata = trdata.flow_from_directory(directory="/home-1/aeshak1@jhu.edu/scratch/full-resolution/training", target_size=(2900,4000), batch_size=1,color_mode='grayscale')


with central_storage_strategy.scope():
        model_0 = Sequential()


        model_0.add(Conv2D(filters=16, kernel_size=(5, 5), padding='same', strides=(3,3), activation='relu', input_shape=(2900,4000,1)))
        model_0.add(BatchNormalization())
        model_0.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', strides=(2,2), activation='relu'))
        model_0.add(BatchNormalization())
        model_0.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))


        model_0.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
        model_0.add(BatchNormalization())
        model_0.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
        model_0.add(BatchNormalization())
        model_0.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

        model_0.add(Conv2D(64, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu'))
        model_0.add(BatchNormalization())
        model_0.add(Conv2D(64, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu'))
        model_0.add(BatchNormalization())
        model_0.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

        model_0.add(Conv2D(128, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu'))
        model_0.add(BatchNormalization())
        model_0.add(Conv2D(128, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu'))
        model_0.add(BatchNormalization())
        #model_0.add(Conv2D(128, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu'))
        model_0.add(MaxPooling2D((2, 2), strides=(2,2)))

        model_0.add(Conv2D(256, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu'))
        model_0.add(BatchNormalization())
        model_0.add(Conv2D(256, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu'))
        model_0.add(BatchNormalization())
        #model_0.add(Conv2D(256, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu'))
        model_0.add(MaxPooling2D((2, 2), strides=(2,2)))


        #model_0.add(Conv2DTranspose(128, (2,2), activation='relu', padding='same'))
        #model_0.add(Conv2D(filters=128, kernel_size=(5, 5), padding='same', strides=(3,3), activation='relu'))
        #model_0.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=(2,2), activation='relu'))
        #model_0.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', strides=(2,2), activation = 'relu'))
        #model_0.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))


        #model_0.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
        #model_0.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
        #model_0.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))


        #model_0.add(Conv2D(32, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu'))
        #model_0.add(Conv2D(32, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu'))
        #model_0.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

        #model_0.add(Conv2D(16, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu'))
        #model_0.add(Conv2D(16, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu'))
        #model_0.add(MaxPooling2D((2, 2), strides=(2,2)))

        #model_0.add(Conv2D(512, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
        #model_0.add(Conv2D(512, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
        #model_0.add(Conv2D(512, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
        #model_0.add(MaxPooling2D((2,2)))

        #model_0.add(Conv2D(1024, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
        #model_0.add(Conv2D(1024, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
        #model_0.add(Conv2D(1024, kernel_size=(3, 3), padding='same', strides=(1,1), activation='relu'))
        #model_0.add(MaxPooling2D((2,2)))

        model_0.add(Flatten())

        #model_0.add(Dense(1024, activation='relu'))
        model_0.add(Dense(2, activation='relu'))
        model_0.add(Dense(2, activation='sigmoid'))

        es = EarlyStopping(monitor='loss', mode='min', patience=20)

        model_0.summary()


        checkpoint = ModelCheckpoint("best_model2.hdf5", monitor='accuracy', verbose=1, save_best_only=True, mode='auto', period=1)
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
                                              