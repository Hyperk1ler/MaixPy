import cv2,imutils
# 读取图片
# 图片加载函数 .imread(filename, flags = None)
# filename 表示图像完整路径,图片需以代码所处位置为根目录
# 读取的文件返回cv::Mat对象,即图像的数据矩阵
# 在此项目中，如果将‘cat.jpg’或'lena_img.png'放在Flie文件夹中，则会报错
image = cv2.imread('cat.jpg', cv2.IMREAD_GRAYSCALE)
# flags为图像读取的标识，如 cv2.IMREAD_GRAYSCALE 则以灰度图的方式读取
# 常用的flags有 cv2.IMREAD_COLOR cv2.IMREAD_UNCHANGED
#生成灰度图
imageGrey = cv2.imread('lena_img.png', 0)
# 显示图片
# 显示函数 .imshow(winname, mat)
# winname 即 window name 窗口名称，为字符串类型
# mat 为图像对象 类型为numpy中的ndarray类型(可用imutils模块进行改变图像显示大小的操作)
cv2.imshow('Image', image)

print(image.shape)
resized = imutils.resize(image,height=200)
cv2.imshow('resized',resized)

cv2.imshow('ImageGrey', imageGrey)
# 与.destroyWindow()函数共用，实现窗口的创建与销毁
if cv2.waitKey(0) == ord('a'):
    cv2.destroyWindow('ImageGrey')
# .waitKey([delay])
# 在 delay 大于 0 时，表示延迟delay毫秒后 触发True
# 在 delay 其他情况时，等待键盘敲击，接收到键盘敲击后 返回 键盘值 (== True)
# 保存图片
cv2.imwrite('Copy.jpg', imageGrey)
# 检查图像保存到本地 .imwrite(image_filename, image)
# image_filename 保存的图像名称，字符串类型
# image 图像对象 ndarray类型
cap = cv2.VideoCapture(1)
# VideoCapture对象，参数为设备索引或者视频文件名称

while True:
    ret, frame = cap.read()
    # 返回一个布尔值和一个图像矩阵
    # 可以通过判断布尔值的方式判定视频是否结束
    b, g, r,= cv2.split(frame)
    #.split()方法可以将图像按b,g,r三个通道分离
    #也可以直接通过numpy引索的方式进行分离（一般来说会用这个，因为split()比较耗时）
    cv2.imshow("Blue", b)
    cv2.imshow("Green", g)
    cv2.imshow("Red", r)
    #cv2.imshow('Camera',frame)
    merge = cv2.merge([b, g, r])
    #.merge()方法可以将b g r三个通道合成一个图像
    cv2.imshow("merged", merge)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
# 关闭视频文件/相机设备
# 可以通过.get(propId)访问视频某些功能，propId为0-18的数字，每个数字表示视频的属性
# 其中一些值可以通过.set(propId, value)方式修改，value为修改过后的值

# .cvtColor(src, code, dst = none, dstcn = none)方法可以转换色彩空间
# code 表示目标色彩空间
cv2.waitKey(0)
cv2.destroyAllWindows()