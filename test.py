import tensorflow as tf
import os
import numpy as np
import random
import cv2 as cv



##import matplotlib.pyplot as plt

##from skimage.io import imread,imshow

model = tf.keras.models.load_model('Modelv1')

data = np.load('data.npz')
X_train = data['X_train']
X_test = data['X_test']
Y_test = data['Y_test']

preds_test = model.predict(X_test)
preds_test = (preds_test > 0.5).astype(np.uint8)




ix = random.randint(0, len(X_test))




# img = cv.cvtColor(Y_test[ix], cv.COLOR_BGR2GRAY)
# cv.imshow('Y',img) 
# cv.waitKey(0)
	





cv.imshow('X',np.array(X_test[ix]).astype('uint8'))
cv.waitKey(0)

cv.imshow('XP',np.array(preds_test[ix]).astype('uint8'))
cv.waitKey(0)
cv.destroyAllWindows()







cv.imshow('preds',preds_test[ix])
cv.waitKey(0)
cv.destroyAllWindows()

# cv.imshow(np.mat(np.squeeze(Y_test[ix])))

# cv.imshow(np.mat(np.squeeze(preds_test[ix])))



