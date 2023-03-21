import tensorflow as tf
import os
import numpy as np

from skimage.io import imread,imshow

model = tf.keras.models.load_model('Modelv1')

model.predict
