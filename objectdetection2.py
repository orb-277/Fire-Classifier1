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
def count_yellow(hsv):
     # Define the lower and upper bounds of the yellow color in HSV
    lower_yellow = (20, 200, 200)
    upper_yellow = (30, 255, 255)

    # Threshold the image to extract yellow regions
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Count the number of yellow pixels in the image
    num_yellow_pixels = cv2.countNonZero(mask)

    # Display the number of yellow pixels
    print("Number of yellow pixels:", num_yellow_pixels)
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
            # cv_image = image = cv2.imread('image1.jpg')
            frame = cv2.resize(cv_image, (960, 540))
            rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            #blur = cv2.GaussianBlur(frame, (21, 21), 0)
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            count_yellow(hsv)
            avg_brightness = np.mean(hsv[:,:,2])
            bright_pixel_count = count_bright_pixels(hsv, 254)

            ratio = (avg_brightness / bright_pixel_count)
            upper_v=255
            print(ratio,i)
            #print(brightness, i)
            # lower = [0, 74, 200]
            # upper = [35, 255, 255]
            lower = [10, 70, 50]
            upper = [30, 255, 255]
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")

            mask = cv2.inRange(hsv, lower, upper)

            output = cv2.bitwise_and(frame, hsv, mask=mask)
            # mask2 = cv2.inRange(hsv, low_green, high_green)
            # # inverse mask
            # # mask2 = 255-mask2
            # res = cv2.bitwise_and(output, output, mask=mask2)
            # height, width, _ = output.shape



            # ret,thresh = cv2.threshold(mask, 40, 255, 0)
            # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # if len(contours) != 0:
            #     # draw in blue the contours that were founded
            #     c = max(contours, key = cv2.contourArea)
            #     cv2.drawContours(output, c, -1, 0, 3)

            # #cv2.fillPoly(output, pts =[c], color=(255,255,255))
            # for i in range(height):
            #     for j in range(width):
            #         # img[i, j] is the RGB pixel at position (i, j)
            #         # check if it's [0, 0, 0] and replace with [255, 255, 255] if so
            #         if output[i, j].sum() >0 and output[i, j].sum() <765:
            #             output[i, j] = [0,0,0]  
                
            cv2.imwrite(filename1 + str(i)+ ".jpg", output)
            i=i+1


   
# cv2.imshow("window",output)
# cv2.waitKey()
# cv2.imwrite('filename2.jpg', output)