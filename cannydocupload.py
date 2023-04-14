import numpy as np
import cv2
import base64


def order_points(pts):
    '''Rearrange coordinates to order:
      top-left, top-right, bottom-right, bottom-left'''
    rect = np.zeros((4, 2), dtype='float32')
    pts = np.array(pts)
    s = pts.sum(axis=1)
    # Top-left point will have the smallest sum.
    rect[0] = pts[np.argmin(s)]
    # Bottom-right point will have the largest sum.
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    # Top-right point will have the smallest difference.
    rect[1] = pts[np.argmin(diff)]
    # Bottom-left will have the largest difference.
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect.astype('int').tolist()


def automatic_brightness_and_contrast(image, clip_hist_percent=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha


    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)
# def four_point_transform(image, pts):
#     # obtain a consistent order of the points and unpack them
#     # individually
#     rect = order_points(pts)
#     (tl, tr, br, bl) = rect
#
#     # compute the width of the new image, which will be the
#     # maximum distance between bottom-right and bottom-left
#     # x-coordiates or the top-right and top-left x-coordinates
#     widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
#     widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
#     maxWidth = max(int(widthA), int(widthB))
#
#     # compute the height of the new image, which will be the
#     # maximum distance between the top-right and bottom-right
#     # y-coordinates or the top-left and bottom-left y-coordinates
#     heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
#     heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
#     maxHeight = max(int(heightA), int(heightB))
#
#     # now that we have the dimensions of the new image, construct
#     # the set of destination points to obtain a "birds eye view",
#     # (i.e. top-down view) of the image, again specifying points
#     # in the top-left, top-right, bottom-right, and bottom-left
#     # order
#     dst = np.array([
#         [0, 0],
#         [maxWidth - 1, 0],
#         [maxWidth - 1, maxHeight - 1],
#         [0, maxHeight - 1]], dtype = "float32")
#
#     # compute the perspective transform matrix and then apply it
#     M = cv2.getPerspectiveTransform(rect, dst)
#     warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
#
#     # return the warped image
#     return warped
#



# loading image


def scan(image):
    # image = cv2.imread("C:/Users/Aryan Dande/Downloads/proper.jpeg")
    # Compute the ratio of the old height to the new height, clone it,
    # and resize it easier for compute and viewing
    orig = image.copy()
    orig, alpha, beta = automatic_brightness_and_contrast(orig.copy())
    ### convert the image to grayscale, blur it, and find edges in the image
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blu the image for better edge detection

    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    edged = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)  # Canny
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # apply the dilation operation to the edged image
    dilate = cv2.dilate(edged, kernel, iterations=2)

    # finding the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cnts, heirarchy = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(image, cnts, -1, (0, 255, 0), 2)
    cv2.imwrite("contour.jpg", image)
    # Taking only the top 5 contours by Area
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    # looping over the contours
    corners = cnts[0]
    print(corners)
    corners = sorted(np.concatenate(corners).tolist())
    # For 4 corner points being detected.
    corners = order_points(corners)

    print(corners)

    top_left_x = min([corners[0][0], corners[1][0], corners[2][0], corners[3][0]])
    top_left_y = min([corners[0][1], corners[1][1], corners[2][1], corners[3][1]])
    bot_right_x = max([corners[0][0], corners[1][0], corners[2][0], corners[3][0]])
    bot_right_y = max([corners[0][1], corners[1][1], corners[2][1], corners[3][1]])
    img = orig[top_left_y:bot_right_y + 1, top_left_x:bot_right_x + 1]
    cv2.imshow("Boundary", img)
    cv2.imwrite("crop.jpg",img)
    #
    #    cv2.waitKey(0)
    return img


def decode_from_bin(bin_data):
    bin_data = base64.b64decode(bin_data)
    image = np.asarray(bytearray(bin_data), dtype=np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return img


###input base64 encoded value of your image here
image =""
temp=scan(decode_from_bin(image))

_, im_arr = cv2.imencode('.jpg', temp)  # im_arr: image in Numpy one-dim array format.
im_bytes = im_arr.tobytes()
im_b64 = base64.b64encode(im_bytes).decode('UTF-8')
print(im_b64)


