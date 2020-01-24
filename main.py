import numpy as np
import cv2
import numpy as np
import cv2
from matplotlib import pyplot as plt


first_frame = cv2.imread("input.png", cv2.IMREAD_GRAYSCALE)
first_frame = cv2.resize(first_frame, (600, 400))


# global thresholding
ret1,th1 = cv2.threshold(first_frame ,220, 10,cv2.THRESH_BINARY)

# Otsu's thresholding
ret2,th2 = cv2.threshold(first_frame,0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(first_frame,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

contours, hieracchy = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img = cv2.cvtColor(first_frame, cv2.COLOR_GRAY2BGR)
for contour in contours:
    img_contoured = cv2.drawContours(img, contour, -1, (0, 0, 0), 3)


cv2.imshow("contoured image", img_contoured)
cv2.waitKey()

