# GAN

## 01 Basic GAN
 * [Basic GAN](https://github.com/stesha2016/GAN/blob/master/tensorflow_GAN_basic.ipynb)
 * 最基础的GAN网络，全部使用FC进行网络连接
 * D网是［None, 784］ -> [1]
   G网是［None, 100］ -> [784]
   G网的input是随机生成的100个数字，output是通过G网后生成的784（28＊28）的图片G_fake。loss为将G_fake送入D网得到的数字与label为1之间的差值
   D网的input是G_fake和mnist中的图片D_real，output是一个评估是否为真实图片的数字。loss有两部分，第一部分G_fake生成的数字与label为0之间的差值，另一部分是D_real生成的数字与label为1之间的差值，两部分相加。
 * D和G同时训练20000次，效果并不理想。
