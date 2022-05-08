import cv2
import numpy as np
from utils import *

def fun(x):
    pass


img_path = "maxresdefault.jpg"

cv2.namedWindow("trackbar")
cv2.createTrackbar("thresh", "trackbar", 50, 255, fun)
cv2.createTrackbar("blur", "trackbar", 5, 10, fun)
cv2.createTrackbar("low", "trackbar", 0, 50000, fun)
cv2.createTrackbar("high", "trackbar", 50, 50000, fun)

while True:
    accepted = []

    # Update the image
    img = cv2.imread(img_path)

    # Resize the image into half size
    img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))

    # Original Image
    orig = img.copy()

    # Convert it to greyscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur the image
    b = int(cv2.getTrackbarPos('blur', 'trackbar'))
    blur = cv2.blur(gray, (b, b))

    # Threshold the image
    r = int(cv2.getTrackbarPos('thresh', 'trackbar'))
    ret, thresh = cv2.threshold(blur, r, 255, 0)

    # Extracting the canny area
    canny = cv2.Canny(thresh, 100, 200)

    # Find the contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    low = int(cv2.getTrackbarPos('low', 'trackbar'))
    high = int(cv2.getTrackbarPos('high', 'trackbar'))

    # For each contour, find the convex hull and draw it on the original image.
    for i in range(len(contours)):
        if low < cv2.contourArea(contours[i]) < high:
            hull = cv2.convexHull(contours[i])
            cv2.drawContours(img, [hull], -1, (255, 0, 0), 2)
            accepted = [hull]

    # Background
    bg = np.zeros(orig.shape, np.uint8)
    bg = cv2.drawContours(bg, accepted, -1, (255, 255, 255), -1)

    # Foreground
    fg = cv2.bitwise_not(bg)

    # Blur the Background
    bg = cv2.bitwise_and(bg, orig)

    # Merge Both blur background and good foreground
    fg = cv2.bitwise_and(fg, cv2.blur(orig, (10, 10)))
    res = cv2.bitwise_or(fg, bg)

    imageArray = ([orig, gray, blur, thresh],
                  [canny, img, bg, res])

    # LABELS FOR DISPLAY
    lables = [["Original", "Gray", "Gray Blur", "Binary"],
              ["Threshold", "Edged", "Extracted Contour", "Result"]]

    # Result images in subplot
    stackedImage = stackImages(imageArray, 0.75, lables)

    stackedImage = cv2.resize(stackedImage, (stackedImage.shape[1] // 2, stackedImage.shape[0] // 2))

    # Display the image
    cv2.imshow("Result", stackedImage)
    key = cv2.waitKey(1)

    if key == ord('q'):
        cv2.imwrite("Result.jpg", res)
        break

cv2.destroyAllWindows()