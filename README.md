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

## 03 WGAN（Wasserstein GAN）
 * [WGAN](https://github.com/stesha2016/GAN/blob/master/tensorflow_WGAN_CIFAR-10_03.ipynb)
 * WGAN 作者通过数学推导，证明了GAN的缺陷，然后针对缺陷进行了改进，网络模型基本不变
 * 关键点：
   1. D网最后一层去掉sigmoid
   2. D网和G网的loss不取log
   3. 每次更新D网的参数后，把他们的绝对值截断到不超过一个固定常数c
   4. 不用Adam，用RMSProp或者SGD
 * 优点就是明显网络稳定很多，相对DCGAN更容易收敛，一般不会出现GAN网D_loss降到特别小而无法约束G网的情况
 * 缺点：对D网的参数范围进行限制后，很容易训练出D网的参数就集中在c值或者-c值上，而不是在-c到c之间，针对这点的改进就是WGAN-GP
 * 效果：
 
 ![初始图片](https://github.com/stesha2016/GAN/blob/master/image/WGAN-02.png)
 ![WGAN效果](https://github.com/stesha2016/GAN/blob/master/image/WGAN-01.png)
 
## 04 WGAN-GP
 * [WGAN-GP](https://github.com/stesha2016/GAN/blob/master/tensorflow_WGANGP_ANIME_04.ipynb)
 * 针对WGAN的D网训练出来的参数很容易集中在c值或者-c值上的缺点，而出现了WGAN-GP。取消了D网参数的截断，而对D网的loss增加了一个惩罚值
 * 关键点：
   1. 对D网的loss增加了一个penalty的值
      mix = real + epsilon*(fake-real)
      D_mix = discriminator(mix)
      grad = tf.gradients(D_mix, mix)[0]
      slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3]))
      penalty = tf.reduce_mean(tf.square(slopes - 1))
      D_loss = D_loss_fake - D_loss_real + 10*penalty
   2. D网不使用batch normalization
   3. 调试下来，不使用bias效果会更好
 * 有些动漫人物的脸部生成的效果很不错：
  ![图片1](https://github.com/stesha2016/GAN/blob/master/image/wgan-gp1.png)
  ![图片2](https://github.com/stesha2016/GAN/blob/master/image/wgan-gp2.png)
