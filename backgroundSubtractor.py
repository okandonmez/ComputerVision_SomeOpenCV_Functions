import cv2

image = cv2.imread("image.jpg")

backSub = cv2.BackgroundSubtractorMOG2()
fgMask = backSub.apply(image)

cv2.imshow('FG Mask', fgMask)

cv2.waitKey()
