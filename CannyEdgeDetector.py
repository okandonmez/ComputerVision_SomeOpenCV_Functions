import cv2

img = cv2.imread('image.jpg',0)
edges = cv2.Canny(img,100,200)

cv2.imshow("canny edge detection", edges)
cv2.imshow("original_image", img)
cv2.waitKey()
