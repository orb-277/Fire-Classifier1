import tensorflow as tf
import os
import numpy as np
import random
import cv2 as cv



import matplotlib.pyplot as plt

##from skimage.io import imread,imshow

model = tf.keras.models.load_model('Modelv1',compile=False)

data = np.load('data.npz')
X_train = data['X_train']
X_test = data['X_test']
Y_test = data['Y_test']

preds_test = model.predict(X_test)
#preds_test = (preds_test > 0.5).astype(np.uint8)

preds_test = (preds_test > 0.5).astype(np.uint8)
1 in preds_test[0]

ix = random.randint(0, len(X_test))
img = cv.cvtColor(X_test[1],cv.COLOR_BGR2GRAY)
img2=np.array(preds_test[1]).astype('uint8')

# img2 = img2.transpose(2,0,1).reshape(128,-1)
for i in range(img2.shape[0]):
    for j in range(img2.shape[1]):
        if img2[i][j] == 1:
            img2[i][j] = 255

cv.imshow('X',img)
cv.waitKey(0)
# cv.imshow('Y',np.array(Y_test[ix]).astype('uint8'))
cv.imshow('Pred',img2)

contours, hierarchy = cv.findContours(img2,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

c = contours
c = list(c[5])
c_sort_by_y = sorted(c,key = lambda x : x[0][1])
c_sort_by_x = sorted(c,key = lambda x : x[0][0])
c_sort_by_y


def getBaseWidthAndHeight(arr:list):
    min_y = arr[len(arr)-1][0][1]
    max_y = arr[0][0][1]
    range = (min_y - max_y)/10
    min_x = arr[len(arr)-1][0][0]
    max_x = arr[len(arr)-1][0][0]
    #print('hehe',max_x,min_x,range,max_y,min_y,min_y-range)

    
    i = len(arr)-1
    while i >= 0:
        if(arr[i][0][1] < int(min_y-range)):
            break
        if(arr[i][0][0] > max_x):
            max_x = arr[i][0][0]
        if(arr[i][0][0] < min_x):
            min_x = arr[i][0][0]
        i = i -1

    return max_x - min_x,min_y-max_y

# getBaseWidth_Height(c_sort_by_y)

def getSymmetryScore(arr:list):
    min_y = arr[len(arr)-1][0][1]
    max_y = arr[0][0][1]




