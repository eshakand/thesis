import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans
import fnmatch

model = tf.keras.models.load_model('./unet_membrane.hdf5')

trdata = ImageDataGenerator()


def testGenerator(test_path, target_size = (256,256),flag_multi_class = False,as_gray = True):
    for root, dirnames, filenames in os.walk(test_path):
        for filename in fnmatch.filter(filenames, '*.png'):
            filepath = os.path.join(root, filename)
            img = io.imread(filepath)
            img = img / 255
            img = trans.resize(img,target_size)
            img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
            img = np.reshape(img,(1,)+img.shape)
            yield img

def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255

def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)


testGene = testGenerator("../first training/cnn_dataset_test")
results = model.predict_generator(testGene, verbose=1)
saveResult("./test",results)



