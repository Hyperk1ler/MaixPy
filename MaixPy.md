# MaixCAM

## 	第一天

### 		常见模块的使用

#### 			屏幕 display模块

使用时，导入、创建对象再进行操作。

```python
from maix import display	#导入display模块

disp = display.Display()	#创建Display对象

disp.show(img)
```

img对象为maix.image.Image对象 通过camera模块的read获取，通过image模块的load加载文件系统中的图像，通过image模块的Image类创建空白图像

```python
from maix import image, display

disp = display.Display()
img = image.load("/root/dog.jpg")		#load表示由文件系统加载，与其相对有.save()方法，参数一致
disp.show(img)
```

显示文字：

```python
from maix import image, display

disp = display.Display()
img = image.Image(320, 240)
img.draw_rect(0, 0, disp.width(), disp.height(), color=image.Color.from_rgb(255, 0, 0), thickness=-1)		#此即画框 参数为 x, y, w, h, color, thickness 
img.draw_rect(10, 10, 100, 100, color=image.Color.from_rgb(255, 0, 0))
img.draw_string(10, 10, "Hello MaixPy!", color=image.Color.from_rgb(255, 255, 255))
					#写字符串 参数为x, y, text, color
disp.show(img)
```

摄像头读取图像并显示：

```python
from maix import camera, display, app

disp = display.Display()
cam = camera.Camera(320, 240)
while not app.need_exit():				#maix中自带的结束程序标志位
    								  #程序其他地方调用app.set_exit_flag()方法后退出循环
    img = cam.read()
    disp.show(img)
```

背光亮度调整：

```python
disp.set_backlight(50)
#参数即亮度百分比，0-100
#板载文件中设置了100时仅为全部亮度的50%，如想获得更大的亮度
#需将board文件中disp_max_backlight = 50调整为更大的值
```

显示到MaixVision:

```Python
#在调用show方法时，将自动压缩图像并发送到MaixVision
#也可以不通过初始化屏幕的方式，直接调用display对象的send_to_maixvision方法发送图像到MaixVision

from maix import image,display

img = image.Image(320, 240)
disp = display.Display()

img.draw_rect(0, 0, img.width(), img.height(), color=image.Color.from_rgb(255, 0, 0), thickness=-1)
img.draw_rect(10, 10, 100, 100, color=image.Color.from_rgb(255, 0, 0))
img.draw_string(10, 10, "Hello MaixPy!", color=image.Color.from_rgb(255, 255, 255))
display.send_to_maixvision(img)
```

#### 			摄像头 camera模块

同样进行导入、创建对象再使用摄像头的方式

```python
from maix import camera

cam = camera.Camera(640, 480)
#其格式为 .Camera(width, height, format, device, fps, buff_num, open, raw)
#其中format 用于指定摄像头的输出格式，默认为FMT_RGB888(若为FMT_GRAYSCALE则输出灰度图)
#fps 用于指定帧率最大值，默认为-1 即自动调整最大值
while 1:
    img = cam.read()
    print(img)
```

与OpenMV相同，MaixCAM也需进行畸变矫正或其他处理，其方式也与OpenMV大致相同

```Python
from maix import camera, display,app,time

cam = camera.Camera(320, 240)
disp = display.Display()
while not app.need_exit():
    t = time.ticks_ms()
    img = cam.read() 
    img = img.lens_corr(strength=1.5)	#调整strength的值直到画面不再畸变
    disp.show(img)						
```

```Python
cam = camera.Camera(640, 480) 
cam.skip_frames(30)           # 跳过开头的30帧
```

```PYthon
from maix import camera, display
#显示摄像头获取的图像
cam = camera.Camera(640, 480)
disp = display.Display()

while 1:
    img = cam.read()
    disp.show(img)
```

设置摄像头参数，基本也与OpenMV相似

```Python
#如要切换回自动曝光模式需运行cam.exp_mode(0)
cam = camera.Camera()
cam.exposure(1000)		#设置曝光，在设置数值后，摄像头切换到手动曝光模式

cam = camera.Camera()
cam.gain(100)			#设置增益，在设置数值后，摄像头切换到手动曝光模式

cam = camera.Camera()
cam.awb_mode(1)			# 0,开启白平衡;1,关闭白平衡

cam = camera.Camera()
cam.luma(50)		    # 设置亮度，范围[0, 100]
cam.constrast(50)		# 设置对比度，范围[0, 100]
cam.saturation(50)		# 设置饱和度，范围[0, 100]

cam = camera.Camera(width=640, height=480)		#更改图片长宽
#or
cam = camera.Camera()
cam.set_resolution(width=640, height=480)
```

#### 图像基础操作

MaixPy中对图像的操作依赖基础图像模块image。

一般常用的格式有

image.Format.FMT_RGB888 / 

image.Format.FMT_RGBA8888 / 

image.Format.FMT_GRAYSCALE / 

image.Format.FMT_BGR888

以RGB888为例，其在内存中为RGB packed排列，即：

像素1 红色, 像素1 绿色, 像素1 蓝色, 像素2 红色, 像素2 绿色, 像素2 蓝色, ... 依次排列

基本操作与OpenMV相似，且大部分在上述代码块中都有展示。其余未展示的方法：

```Python
from maix import image

img = image.Image(320, 240, image.Format.FMT_RGB888)
img.draw_line(x1, y1, x2, y2, color)					#画线
img.draw_circle(x, y, r, color)							#画圆
img_new = img.resize(w, h)								#缩放图像，返回一个新的图像对象
img_new = img.crop(x1, y1, x2, y2)						#剪裁图像，返回一个新的图像对象
img_new = img.rotate(r)									#旋转图像，返回一个新的图像对象
img_new = img.copy()									#拷贝图像
img_new = img.affine([(10, 10), (100, 10), (10, 100)], [(10, 10), (100, 20), (20, 100)])
#仿射变换，返回一个新的图像对象
keypoints = [10, 10, 100, 10, 10, 100]
img.draw_keypoints(keypoints, image.Color.from_rgb(255, 0, 0), size=10, thickness=1, fill=False)		#画关键点，在坐标(10, 10)、(100, 10)、(10, 100)画三个红色的关键点
```

#### 触摸屏 Touchscreen模块

```Python
from maix import touchscreen, app, time

ts = touchscreen.TouchScreen()

pressed_already = False
last_x = 0
last_y = 0
last_pressed = False
while not app.need_exit():
    x, y, pressed = ts.read()
    if x != last_x or y != last_y or pressed != last_pressed:
        print(x, y, pressed)
        last_x = x
        last_y = y
        last_pressed = pressed
    if pressed:
        pressed_already = True
    else:
        if pressed_already:
            print(f"clicked, x: {x}, y: {y}")
            pressed_already = False
    time.sleep_ms(1)  # sleep some time to free some CPU usage

```

