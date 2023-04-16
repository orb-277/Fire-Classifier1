import cv2
import numpy as np

# Load the image
img = cv2.imread('fire.130.png')

# Convert the image to HSV format
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define the lower and upper bounds for yellow, orange, reddish orange, and dark red colors in HSV
lower_yellow = np.array([20, 100, 200])
upper_yellow = np.array([30, 255, 255])
lower_bright_yellow = np.array([20, 255, 255])
upper_bright_yellow = np.array([25, 40, 255])
lower_orange = np.array([5, 200, 200])
upper_orange = np.array([10, 255, 255])
lower_reddish_orange = np.array([5, 200, 200])
upper_reddish_orange = np.array([15, 255, 255])
lower_dark_red = np.array([160, 100, 100])
upper_dark_red = np.array([180, 200, 200])
lower_white = np.array([0,0,200])
upper_white = np.array([180,50,255])
# Threshold the image to extract yellow, orange, reddish orange, and dark red regions
mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
mask_reddish_orange = cv2.inRange(hsv, lower_reddish_orange, upper_reddish_orange)
mask_dark_red = cv2.inRange(hsv, lower_dark_red, upper_dark_red)
mask_white = cv2.inRange(hsv, lower_white, upper_white)
# Threshold the image to extract bright yellow regions
#mask_bright_yellow = cv2.inRange(hsv, lower_bright_yellow, upper_bright_yellow)

# Combine the masks for yellow, orange, reddish orange, and dark red regions
mask = cv2.bitwise_or(mask_yellow, mask_orange)
mask = cv2.bitwise_or(mask, mask_reddish_orange)
mask = cv2.bitwise_or(mask, mask_dark_red)
#mask = cv2.bitwise_or(mask, mask_bright_yellow)

# Remove white regions from the mask
mask = cv2.bitwise_not(mask_white, mask_white, mask)
# Find the contours of the regions
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a black image of the same size as the original image
result = np.zeros_like(img)

# Draw the detected regions as white on the result image
cv2.drawContours(result, contours, -1, (255, 255, 255), -1)

# Set the first and last rows of the result image to all zeros
result[0,:] = 0
result[-1,:] = 0

# Display the resulting image
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
