import tensorflow as tf
import os
import numpy as np

import matplotlib.pyplot as plt

from skimage.io import imread,imshow

model = tf.keras.models.load_model('Modelv1')

data = np.load('data.npz')
X_test = data['X_test']
Y_test = data['Y_test']



imshow(X_test[0])
plt.show()
imshow(np.squeeze(Y_test[0]))
plt.show()
imshow(np.squeeze(model.predict(X_test[0])))
plt.show()