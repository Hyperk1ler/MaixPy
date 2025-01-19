import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('cat.jpg')
image_add = image + 10                  #此即numpy库的加法
image_add2 = cv2.add(image,image_add)   #此即opencv的add()函数
                                        #即 运算结果大于255，取255
img_photo = cv2.imread('james.jpg')

print(image[0:4, :, 0])
print(image_add[0:4, :, 0])
print(image_add2[0:4, :, 0])

img = cv2.imread('logo1.jpg')
test = img
im = cv2.threshold(image,155,255,cv2.THRESH_BINARY)[1]

cv2.imshow('image',im)
#opencv算法
img_test = cv2.add(img,test)

#numpy算法
img_test1 = img + test

all_test = np.column_stack((img, img_test, img_test1))

cv2.imshow('img, img_test, img_test1',all_test)

rec = np.vstack((img, img_test, img_test1))
cv2.imshow('rec',rec)


'''
img_logo = cv2.imread('logo1.jpg')

print(img_logo.shape, img_photo.shape)
# (615, 327, 3) (640, 1024, 3)

rows, cols, channels = img_logo.shape
photo_roi = img_photo[0:rows, 0:cols]

gray_logo = cv2.cvtColor(img_logo, cv2.COLOR_BGR2GRAY)
# 中值滤波
midian_logo = cv2.medianBlur(gray_logo, 5)
# mask_bin  是黑白掩膜
ret, mask_bin = cv2.threshold(gray_logo, 127, 255, cv2.THRESH_BINARY)

# mask_inv 是反色黑白掩膜
mask_inv = cv2.bitwise_not(mask_bin)

# 黑白掩膜 和 大图切割区域 去取和
img_photo_bg_mask = cv2.bitwise_and(photo_roi, photo_roi, mask=mask_bin)

# 反色黑白掩膜 和 logo 取和
img2_photo_fg_mask = cv2.bitwise_and(img_logo, img_logo, mask=mask_inv)

dst = cv2.add(img_photo_bg_mask, img2_photo_fg_mask)

img_photo[0:rows, 0:cols] = dst

cv2.imshow("mask_bin", mask_bin)
cv2.imshow("mask_inv", mask_inv)
cv2.imshow("img_photo_bg_mask", img_photo_bg_mask)
cv2.imshow("img2_photo_fg_mask", img2_photo_fg_mask)
cv2.imshow("img_photo", img_photo)
cv2.imshow("midian", midian_logo)
'''


img = cv2.imread('License-plate.jpg')
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# 设127 为全局阈值
ret1,th1 = cv2.threshold(gray_img,127,255,cv2.THRESH_BINARY)

# Otsu 滤波
ret2,th2 = cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print(ret2)

# 先使用一个 5x5 的高斯核除去噪音，然后再使用 Otsu 二值化
blur = cv2.GaussianBlur(gray_img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imshow("iamg",gray_img)
cv2.imshow("blur",blur)
cv2.imshow("img",img)
cv2.imshow("th1",th1)
cv2.imshow("th2",th2)
cv2.imshow("th3",th3)

cv2.waitKey(0)
cv2.destroyAllWindows()