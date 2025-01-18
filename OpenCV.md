# OpenCV

## 第一天

### 图像基本操作

#### 计算机中的图像

基本都写在代码中了，改天整理笔记

## 第二天

### 图像算术运算 & 图像阈值

#### 图像算术运算

要叠加两张图片，可以用 cv2.add() 函数，相加两幅图片的形状（高度/宽度/通道数）必须相同

```python
#opencv算法
img_test = cv2.add(img,test)

#numpy算法
img_test1 = img + test

all_test = np.column_stack((img, img_test, img_test1))	#用于将一个或者多个一维数组或二维数组沿着列方向堆叠成一个二维数组

cv2.imshow('img, img_test, img_test1',all_test)
```

![image-20250118220741278](C:\Users\Jamuq\AppData\Roaming\Typora\typora-user-images\image-20250118220741278.png)

效果上如上图所示，从左到右依次为 原图、OpenCV算法、NumPy算法

图像融合是指将两张或以上的图像信息融合到一张图像上。

实际上是在图像加法的基础上增加了系数和亮度调节量

![image-20250118221020519](C:\Users\Jamuq\AppData\Roaming\Typora\typora-user-images\image-20250118221020519.png)

OpenCV中的图像混合.addWeighted() 是一种加权的图像相加操作 （所以他依旧需要两个图象相等）

![image-20250118221133774](C:\Users\Jamuq\AppData\Roaming\Typora\typora-user-images\image-20250118221133774.png)

γ即修正值，α，β即图像1，2的权重系数

```Py
import cv2
import numpy as np
 
img1 = cv2.imread('lena_small.jpg')
img2 = cv2.imread('opencv_logo_white.jpg')
# print(img1.shape, img2.shape)  # (187, 186, 3) (184, 193, 3)
img2 = cv2.resize(img2, (186, 187))
# print(img1.shape, img2.shape)
res = cv2.addWeighted(img1, 0.6, img2, 0.4, 0)
cv2.imshow("res", res)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

图像矩阵减法与加法类似

```Py
cv2.subtract(src1, src2, dst = None, mask = None, dtype = None)
#src1:图像矩阵1
#src2:图像矩阵2
#dst：默认选项
#mask：默认选项
#dtype：默认选项
```

按位运算

```py
cv2.bitwise_and()		#按位 与
cv2.bitwise_not()		#按位 或
cv2.bitwise_or()		#按位 非
cv2.bitwise_xor()		#按位 异或
#其参数均为
#src1:图像矩阵1
#src2:图像矩阵2
#dst：默认选项
#mask：默认选项
```

从多值的数字图像中直接提取出目标物体，常用的方法就是设定一个阈值T，用 T 将图像的数据分为两部分：大于 T 的像素群和小于 T 的像素群。这是研究灰度变换的最特殊的方法，称为图像二值化（Binarization）

掩膜（mask）实际上就是遮罩

用一个二值化图片对另一幅图片进行局部遮挡

![image-20250118222603540](C:\Users\Jamuq\AppData\Roaming\Typora\typora-user-images\image-20250118222603540.png)

可以看到 原始数据 按位与 遮罩后 遮罩上0对应的区域被取零，而1对应的区域不变，这就做到了对0所在区域的忽略
