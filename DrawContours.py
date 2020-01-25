import cv2

shape_image = cv2.imread("shapes.jpg", cv2.IMREAD_GRAYSCALE)
shape_image = cv2.resize(shape_image, (600, 400))

# global thresholding
ret1,th1 = cv2.threshold(shape_image ,220, 10,cv2.THRESH_BINARY)

# Otsu's thresholding
ret2,th2 = cv2.threshold(shape_image,0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(shape_image,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# if want the change thresholding method, we can try the th1, th2, th3
contours, hieracchy = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img = cv2.cvtColor(shape_image, cv2.COLOR_GRAY2BGR)

for contour in contours:
    img_contoured = cv2.drawContours(img, contour, -1, (0, 0, 0), 3)

cv2.imshow("contoured image", img_contoured)
cv2.waitKey()

