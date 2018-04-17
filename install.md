# 在win10上安装GPU版本

1.下载并安装CUDA®工具包9.0 **【这个版本必须跟官网上一致】**

https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows

安装好后，将cuda路径名添加到环境变量。



2.安装与CUDA Toolkit 9.0相关的NVIDIA驱动程序



3.下载cuDNN v7.0。**【这个版本也必须跟官网写的一致】**

https://developer.nvidia.com/cudnn

解压文件

1. Copy <installpath>\cuda\bin\cudnn64_7.dll to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin.
2. Copy <installpath>\cuda\ include\cudnn.h to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include.
3. Copy <installpath>\cuda\lib\x64\cudnn.lib to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64.

确保设置了以下环境变量：

```
Variable Name: CUDA_PATH 
Variable Value: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0
```



4.直接pip安装TensorFlow

```
pip install tensorflow-gpu
```



5.验证是否安装成功

```
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```