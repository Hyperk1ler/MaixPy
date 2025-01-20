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

边界填充

对于图像的卷积操作，最边缘的像素一般无法处理，所以卷积核中心倒不了最边缘像素。这就需要先将图像的边界填充，再根据不同的填充算法进行卷积操作，得到的新图像就是填充后的图像。

```py
cv2.copyMakeBorder(src, top, bottom, left, left, borderType, dst = None, value = None)
# 在图像周围创建一个边
# src = 输入图像
# top, bottom, left, right 对应边界的像素数目
# borderType = 添加边界的类型，类型如下
# .BORDER_CONSTANT 添加有颜色的常数值边界，需要参数value
# .BORDER_REFLECT 例 fedcba|abcdefgh|hgfedcb
# .BORDER_REFLECT_101 / .BORDER_DEFAULT 例 gfedcb|abcdefgh|gfedcba
# .BORDER_REPLICATE 例 aaaaaa|abcdefgh|hhhhhhh
# .BORDER_WRAP 例 cdefgh|abcdefgh|abcdefg
```

为了对比各种方法得出的图像的区别 可使用 numpy 库中.vstack()方法或.hstack()方法

也可使用上面用到过的.column_stack()方法，三种方法都有一个共同条件

即 合成的图像大小必须一致，否则会报错

二值化处理

即设阈值 大于阈值的为0（黑色）或255（白色）使图像成为黑白图

阈值可固定，也可自适应阈值，常用的二值化算法即如下图

![image-20250119213505139](C:\Users\Jamuq\AppData\Roaming\Typora\typora-user-images\image-20250119213505139.png)

通过设定一个阈值T，使高于阈值的部分成为白色，低于阈值的部分成为黑色

而阈值，也分为全局阈值或局部阈值

OpenCV中一个简单的全局阈值处理即.threshold()方法

```py
cv2.threshold(src, thresh, maxval, type, dst = None)

# src = 原图像
# thresh = 阈值
# maxval = 当像素值高于（或小于）阈值时应被赋予的新的像素值
# type = 阈值调整方法 具体类型如下
# THRESH_BINARY			超过阈值部分 取最大值 否则取零 即下图的二进制阈值
# THRESH_BINARY_INV		上述方法的翻转 即下图中反二进制阈值
# THRESH_TRUNC			大于阈值部分设为阈值 否则不变 即下图中截断阈值
# THRESH_TOZERO			大于阈值部分不变 否则设为零 即下图中阈值化为0
# THRESH_TOZERO_INV		上述方法的翻转 即下图中反阈值化为0
```

![image-20250119214600086](C:\Users\Jamuq\AppData\Roaming\Typora\typora-user-images\image-20250119214600086.png)

OpenCV中一个自适应阈值处理函数（局部阈值）为.adaptiveThreshold()方法

```py
cv2.adaptiveThreshold(src, macValue, adaptiveMethod, thresholdType, blockSize, C, dst = None)

# src = 原图像 应为灰度图
# maxValue = 当像素值高于（或小于）阈值时应被赋予的新像素值
# adaptiveMethod 为 CV_ADAPTIVE_THRESH_MEAN_C 或 CV_ADAPTIVE_GAUSSIAN_C 中的一个
# _MEAN_C 指邻近区域的均值 _GAUSSIAN_C 指邻近区域的加权和，权重为一个高斯窗口
# thresholdType = 仅有 .THRESH_BINARY 和 .THRESH_BINARY_INV
# blockSize = 规定区域大小（为一个正方形区域）
# C= 为一个常数 对于_MEAN_C 方法 阈值则等于区域均值减去C 对于_GAUSSIAN_C 方法 阈值则等于加权值减去C
```

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/a3647d2ecfe25ba3a51f3739cb73fb3b.jpeg)

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/e17f97b537a94323b0205b478e274db9.png)

Otsu 二值化

使用全局阈值时的阈值，几乎为一个随机值，需要不断尝试不同的值才能逐渐获得一个较优的阈值。

如果在处理一副双峰图像时，应在两个峰间峰谷选一个值作为阈值

而Otsu 二值化即 对一幅双峰图像自动根据直方图计算阈值（）用到的函数为.threshold() 但需多传入一个参数flag cv2.THRESH_OTSU

```py
import cv2

img = cv2.imread('License-plate.jpg')
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# 设127 为全局阈值
ret1,th1 = cv2.threshold(gray_img,127,255,cv2.THRESH_BINARY)

# Otsu 滤波
ret2,th2 = cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print(ret2)

# 先使用一个 5x5 的高斯核除去噪音，然后再使用 Otsu 二值化
blur = cv2.GaussianBlur(gray_img,(5,5),0)	#做一个类似 降锐度 的操作
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imshow("img",img)
cv2.imshow("th1",th1)
cv2.imshow("th2",th2)
cv2.imshow("th3",th3)

cv2.waitKey(0)
cv2.destroyAllWindows()

```

![image-20250119220833763](C:\Users\Jamuq\AppData\Roaming\Typora\typora-user-images\image-20250119220833763.png)

## 第三天

### 图像灰度线性变换与非线性变换（对数变换，伽马变换）

#### 图像灰度化

即将彩色图像转化为灰度图像。彩色图像中每个像素颜色由RGB三个分量决定，每个分量可取值0~255

则一个像素点0~255^3的取值范围。

灰度图像是一种RGB三通道都相同的特殊彩色图像。一个像素点有0~255的取值范围

灰度化的核心思想即为 R = G = B 这个相同的值叫灰度值

一种常见常用的方法是 加权平均灰度处理 且具有一个比较符合的经验值

通过人眼对红、绿、蓝三种颜色的敏感度 分别设置权值为 0.299、0.587、0.114

即可得到较为合理的灰度图像

#### OpenCV中灰度化

直接灰度化，在读取图像时直接将其转化为灰度图

```py
import cv2

img = cv2.imread(photo_file, cv2.IMREAD_GRAYSCALE)
```

先读取后转化为灰度图

```py
import cv2

img = cv2.imread(photo_file)

grey_img = cv2.cvtcolor(img, cv2.COLOR_BGR2GRAY)
```

PIL库中的Image模块

```py
import numpy as np
form PIL import Image

img2 = Image.open(photo_file)
grey_img2 = img.convert('L')
grey_img22 = np.array(grey_img2)
print(type(grey_img22))
```

图像灰度化练习

```py
import cv2

img = cv2.imread('lena.jpg')
res = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('original', img)
cv2.imshow('gray', res)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

#### 基于像素操作的图像灰度化

最大值灰度处理

灰度值即RGB三个分量中最大值，处理过后灰度图亮度很高

```py
import cv2 
import numpy as np 
 
#读取原始图像
img = cv2.imread('irving.jpg')
src = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
#获取图像高度和宽度
height = img.shape[0]
width = img.shape[1]
 
#创建一幅图像
grayimg = np.zeros((height, width, 3), np.uint8)
 
#图像最大值灰度处理
for i in range(height):
    for j in range(width):
        #获取图像R G B最大值
        gray = max(img[i,j][0], img[i,j][1], img[i,j][2])
        #灰度图像素赋值 gray=max(R,G,B)
        grayimg[i,j] = np.uint8(gray)
 
cv2.imshow("src", img)
cv2.imshow("gray", grayimg)
 
cv2.waitKey(0)
cv2.destroyAllWindows()
```

平均灰度处理

灰度值等于三个分量灰度值的求和平均，处理过后图像比较柔和

```py
import cv2 
import numpy as np 
import matplotlib.pyplot as plt
 
#读取原始图像
img = cv2.imread('irving.jpg')
 
src = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
#获取图像高度和宽度
height = img.shape[0]
width = img.shape[1]
 
#创建一幅图像
grayimg = np.zeros((height, width, 3), np.uint8)
 
#图像平均灰度处理方法
for i in range(height):
    for j in range(width):
        #灰度值为RGB三个分量的平均值
        gray = (int(img[i,j][0]) + int(img[i,j][1]) + int(img[i,j][2]))  /  3
        grayimg[i,j] = np.uint8(gray)

cv2.imshow("src", img)
cv2.imshow("gray", grayimg)
 
cv2.waitKey(0)
cv2.destroyAllWindows()
```

图像灰度线性变换

通过建立灰度映射调整原始图像的灰度

![image-20250120222323860](C:\Users\Jamuq\AppData\Roaming\Typora\typora-user-images\image-20250120222323860.png)

Db表示灰度线性变换后的灰度值，Da表示变换前输入图像的灰度值，a/b为f(D)的参数

![image-20250120222435428](C:\Users\Jamuq\AppData\Roaming\Typora\typora-user-images\image-20250120222435428.png)
