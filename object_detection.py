import cv2
import numpy as np
#import pyrealsense2 as rs
moments_array = []
cv_image=np.ndarray(shape=(480, 640, 3),dtype=np.uint8)
mask=np.ndarray(shape=(480, 640),dtype=np.uint8)
depth_image=np.ndarray(shape=(480, 640, 3),dtype=np.uint32)


cv_image = image = cv2.imread('image2.jpg')
   

height, width, channels = cv_image.shape
#print(height,width,channels)
descentre = 160
rows_to_watch = 60
crop_img = cv_image[int((height)/2+descentre):int((height)/2+(descentre+rows_to_watch))][1:width]
#print(crop_img)
hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
upper_red = np.array( [18, 50, 50])
lower_red = np.array([35, 255, 255])

# Threshold the HSV image to get only red colors
mask = cv2.inRange(hsv, lower_red, upper_red)

output = cv2.bitwise_and(cv_image, hsv, mask=mask)
contours,_ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
#cv2.drawContours(cv_image, contours, -1, (0,255,0), 3)
moments_array = []
polygon_list = []
for contour in contours:
    approx = cv2.approxPolyDP(contour,0.001*cv2.arcLength(contour,True),True)
    polygon_list.append(approx)
for polygon in polygon_list:
    cv2.drawContours(cv_image, polygon_list , -1, (255,0,0), 1)
for contour in contours:
    moment = cv2.moments(contour)
    try:
        if (moment['m10']/(moment['m00'] + 1e-5)!=0 and moment['m01']/(moment['m00'] + 1e-5))!=0:
            moments_array.append((moment['m10']/(moment['m00'] + 1e-5), moment['m01']/(moment['m00'] + 1e-5)))
    except ZeroDivisionError as e:
        print(e)
# print(len(moments_array))
# for (index, (x, y)) in enumerate(moments_array):
#     cv2.circle(cv_image, (int(x), int(y)), radius=2, color=(255, 255, 255), thickness=-1)
#     cv2.putText(cv_image, "tomato_{}".format(index), (int(x)-20, int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color=(255, 255, 255), thickness=1)


    
    
cv2.imshow("window",output)
cv2.waitKey()
cv2.imwrite('filename.jpg', output)
    


