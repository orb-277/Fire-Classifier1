#import tensorflow as tf
import os
import numpy as np
import re

from sklearn.model_selection import train_test_split

from skimage.io import imread
from skimage.transform import resize

from tqdm import tqdm 


TRAIN_PATH = 'Image/Fire'
MASK_PATH = 'Segmentation_Mask/Fire'

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)


masked_images = os.listdir(MASK_PATH)
masked_images = sorted_alphanumeric(masked_images)


train_images = os.listdir(TRAIN_PATH)
train_images = sorted_alphanumeric(train_images)


X_labels = np.zeros((500,IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS),dtype=np.uint8)
Y_labels = np.zeros((500,IMG_HEIGHT,IMG_WIDTH,1),dtype=np.bool_)


x = 0

for i in tqdm(range(500)):
    img = imread(TRAIN_PATH+'/'+train_images[i])[:,:,:IMG_CHANNELS]
    img = resize(img,(IMG_HEIGHT,IMG_WIDTH),mode='constant',preserve_range=True)
    X_labels[x] = img
    x += 1



x = 0
for i in tqdm(range(500)):   
    img = imread(MASK_PATH+'/'+masked_images[i])
    img = np.expand_dims(resize(img,(IMG_HEIGHT,IMG_WIDTH),mode='constant',preserve_range=True), axis=-1)
    Y_labels[x] = img
    x += 1



# #print(X_labels)
# print('----------------------------------------------------------------------')
# print(False in Y_labels[0])

X_train,X_test,Y_train,Y_test = train_test_split(X_labels,Y_labels,test_size=0.20, random_state=42)

np.savez('data',X_train=X_train,X_test=X_test,Y_train=Y_train,Y_test=Y_test)




