import tensorflow as tf
import os
import numpy as np
import random

import matplotlib.pyplot as plt

from skimage.io import imread,imshow

model = tf.keras.models.load_model('Modelv1')

data = np.load('data.npz')
X_train = data['X_train']
X_test = data['X_test']
Y_test = data['Y_test']

preds_test = model.predict(X_test, verbose=1)
preds_test = (preds_test > 0.5).astype(np.uint8)




ix = random.randint(0, len(X_test))
imshow(X_test[ix])
plt.show()
imshow(np.squeeze(Y_test[ix]))
plt.show()
imshow(np.squeeze(preds_test[ix]))
plt.show()

print(np.squeeze(preds_test[ix]))
