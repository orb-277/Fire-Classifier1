import cv2
import numpy as np
import os
import re
from skimage.io import imread
from skimage.transform import resize
folder='Fire'
images=[]
filename1 = "segmented_imgs/img" ## something that changes in this loop -> you can set a complete path to manage folders
i=1
def count_bright_pixels(hsv, threshold):
    # convert to HSV color space
    count=0
    for i in hsv[:,:,2]:
        for j in i:
            if j>threshold:
                count+=1
    # create a binary mask for pixels with brightness greater than the threshold
    # mask = hsv[:,:,2] > threshold
    
    # count the number of pixels with brightness greater than the threshold
    #count = np.sum(mask)
    
    return count

low_green = np.array([89, 200, 200])
high_green = np.array([89, 255, 255])
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)


non_masked_images = os.listdir(folder)
non_masked_images = sorted_alphanumeric(non_masked_images)
for img in non_masked_images:
        cv_image = cv2.imread(folder+"/"+img)
        if cv_image is not None:

            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

            # Define the lower and upper bounds for yellow, orange, reddish orange, and dark red colors in HSV
            lower_yellow = np.array([20, 100, 200])
            upper_yellow = np.array([30, 255, 255])
            lower_orange = np.array([5, 50, 200])
            upper_orange = np.array([10, 255, 255])
            lower_reddish_orange = np.array([5, 100, 200])
            upper_reddish_orange = np.array([15, 255, 255])
            lower_dark_red = np.array([160, 100, 100])
            upper_dark_red = np.array([180, 255, 255])
            lower_bright_yellow = np.array([20, 0, 220])
            upper_bright_yellow = np.array([30, 40, 255])
            # Threshold the image to extract yellow, orange, reddish orange, and dark red regions
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
            mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
            mask_reddish_orange = cv2.inRange(hsv, lower_reddish_orange, upper_reddish_orange)
            mask_dark_red = cv2.inRange(hsv, lower_dark_red, upper_dark_red)
            mask_bright_yellow = cv2.inRange(hsv, lower_bright_yellow, upper_bright_yellow)

            # Combine the masks for yellow, orange, reddish orange, and dark red regions
            mask = cv2.bitwise_or(mask_yellow, mask_orange)
            mask = cv2.bitwise_or(mask, mask_reddish_orange)
            mask = cv2.bitwise_or(mask, mask_dark_red)
            mask = cv2.bitwise_or(mask, mask_bright_yellow)

            # Find the contours of the regions
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Create a black image of the same size as the original image
            result = np.zeros_like(cv_image)

            # Draw the detected regions as white on the result image
            cv2.drawContours(result, contours, -1, (255, 255, 255), -1)

            # Set the first and last rows of the result image to all zeros
            result[0,:] = 0
            result[-1,:] = 0

            # Display the resulting image
            # cv2.imshow("Result", result)
           
            
            cv2.imwrite(filename1 + str(i)+ ".jpg", result)
            i=i+1