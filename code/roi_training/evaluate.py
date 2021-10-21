import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, MaxPool2D , Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


model_0 = tf.keras.models.load_model('./vgg19.hdf5')


trdata = ImageDataGenerator()
#
test_images = trdata.flow_from_directory(directory="/home-1/aeshak1@jhu.edu/scratch/roi_training/test",batch_size=2, target_size=(224,224), color_mode='rgb')
test_loss, test_acc = model_0.evaluate(test_images,  batch_size=2, verbose=2)
