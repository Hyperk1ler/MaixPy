import numpy as np
import cv2

img = cv2.imread('lena.jpg')
res = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('original', img)
cv2.imshow('gray', res)

cv2.waitKey(0)
cv2.destroyAllWindows()