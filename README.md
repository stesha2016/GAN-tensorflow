# GAN

## 01 Basic GAN
 * [Basic GAN](https://github.com/stesha2016/GAN/blob/master/tensorflow_GAN_basic.ipynb)
 * 最基础的GAN网络，全部使用FC进行网络连接
 * D网是［None, 784］ -> [1]
   G网是［None, 100］ -> [784]
   G网的input是随机生成的100个数字，output是通过G网后生成的784（28＊28）的图片G_fake。loss为将G_fake送入D网得到的数字与label为1之间的差值
   D网的input是G_fake和mnist中的图片D_real，output是一个评估是否为真实图片的数字。loss有两部分，第一部分G_fake生成的数字与label为0之间的差值，另一部分是D_real生成的数字与label为1之间的差值，两部分相加。
 * D和G同时训练20000次，虽然开始出现数字的雏形，但是效果并不是太理想。
 ![0次迭代](https://github.com/stesha2016/GAN/blob/master/image/01_00.png)
 ![1800次迭代](https://github.com/stesha2016/GAN/blob/master/image/01_01.png)

## 02 DCGAN
 * [DCGAN MNIST](https://github.com/stesha2016/GAN/blob/master/tensorflow_DCGAN_MNIST_02.ipynb)
 * 引入了卷积运算，通过实验得出一套效果相对不错的模型
 * D网［None, 64, 64, 1］ -> [None, 32, 32, 128] -> [None, 16, 16, 256] -> [None, 8, 8, 512] -> [None, 4, 4, 1024] -> [None, 1, 1, 1]
   G网[None, 1, 1, 100] -> [None, 4, 4, 1024] -> [None, 8, 8, 512] -> [None, 16, 16, 256] -> [None, 32, 32, 128] -> [None, 64, 64, 1]
 * 关键点：
   1. 每一层除了out层，都必须加上batch normalization.
   2. Loss函数与原始GAN一样.
   3. 使用lrelu效果会更好.
   4. 训练一次D网，训练两次G网效果会好一些.
 * 缺点：
   1. 训练过程不稳定，很有可能D网的loss会降到特别小，而G网的loss上升，D网约束不了G网了。
   2. 有时候G网生成的图片多样性不足。
   3. 没有一个值衡量结果的好坏，Dloss与Gloss似乎要达到一种平衡才可以。
 * 效果：
 ![15次epoch的效果](https://github.com/stesha2016/GAN/blob/master/image/DCGAN.png)
