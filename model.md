'''  
semantic module  
'at','on','inside_of','under',\ 'wears','interacts_with','hits','holds','plays',  
数据库中主要由这3种关系组成: geometric(几何): 50.9%, possessive(所有格): 40.9% semantic(语义): 8.7%.  
衣服、身体部件大多是所有格关系；家具、建筑大多是几何关系；人大多是语义关系的主语。  
```python
n_classes = 62  
n_relationship = 9  
dict = {}  
for i in range(n_classes-1):  
    for j in range(i+1,n_classes):  
        _ = []  
        for k in range(n_relationship):  
            _.append(TP[(i,j)]/(TP[(i,j)]+FP[(i,j)]))  
        dict[(i,j)] = _  
  
label1, label2 = min(label1, label2), max(label1, label2)  
feed_dict['semantic_logits'] = dict[(label1,label2)]   
   
semantic_logits = tf.placeholder(shape=[None,9], dtype = tf.float32, name = 'semantic_logits')  
```
'''  
Spatial Module    
宾语对主语 主语对联合 宾语对联合    
```python
def box_delta(box1,box2):    
	return [abs(box1[0]-box2[0])/box2[2], abs(box1[1]-box2[1])/box2[1], math.log(box1[2]/box2[2],2), math.log(box1[3]/box2[3],2)]  
def normalized_coordinates(box, image):  
    W, H, C = image.shape  
    return [box[0]/W, box[1]/H, (box[2]-1+box[0])/W, (box[3]-1+box[1])/H, (box[2]*box[3])/(W*H)]  
    
box1, box2 = [x1, y1, w1, h1], [x2, y2, w2, h2]  
boxall = [min(x1,x2), min(y1,y2), max(x1,x2)-min(x1,x2)+1, max(y1,y2)-min(y1,y2)+1]  
  
delta12 = box_delta(box2, box1)  
delta1all = box_delta(box1, boxall)  
delta2all = box_delta(box2, boxall)  
  
c1 = normalized_coordinates(box1)  
c2 = normalized_coordinates(box2)  
  
tmp = delta12 + delta1all + delta2all + c1 + c2  
feed_dict['semantic_logits'] = tmp  
  
spatial_logits = tf.placeholder(shape=[None,22], dtype = tf.float32, name = 'spatial_logits')  
branch = tf_util.conv1d(net, 18, 1, padding='VALID', activation_fn=None, scope='fc2')  
branch = tf_util.conv1d(branch, 9, 1, padding='VALID', activation_fn=None, scope='fc2')  
```
'''  
Visual Module  
fasterrcnn  
![fasterrcnn框图](https://github.com/XiaoPichu/relationship/blob/master/fasterrcnn.png)  
https://www.cnblogs.com/the-home-of-123/p/9747963.html  

featnms  
https://blog.csdn.net/j879159541/article/details/97892863#FasterRCNNNMS_9  
FasterRCNN是个two-stage的方法。
* RPN生成一堆候选框，
* 对候选框进行的后处理，
	* 1.根据score阈值筛选一遍，低于阈值的全部过滤掉。
	* 2.进行IOU-NMS。
	* 3.按照score排序，取分数最高的N个目标。
* 最后将符合要求的候选框送到RCNN里面去回归。  
  
目标框的丢失的：
* RPN没检测到
* 后处理NMS丢失，
* RCNN的时候被过滤掉  
  
解决方式：
* RPN没检测到：
anchor产生方式：每个特征点周围产生9个anchor,分别为3种面积，3种比例。
就是(0,0,15,15）为基准点，有三种比例(0.5,1,2)的框求得基准anchor的坐标后，分别乘(8,16,32）倍的scale，就得到9种anchor的坐标，之后再平移。
anchor的size太大了，网络无法覆盖到这种小样本。于是将anchor的size降低一倍重新训练.
* NMS丢失
先检查了一下RPN的结果，未经过后处理之前，小目标是存在的，经过后处理之后，小目标就被过滤掉了，去掉NMS，让这些小目标进入RCNN，发现RCNN给它们打的分数都很高。那么问题就明确了，是后处理的过程不当导致这些小目标被排除了。

'''  
gumble trick  
https://zhuanlan.zhihu.com/p/50065712  
https://www.zhihu.com/question/62631725  
https://www.jianshu.com/p/9d5a0698f982  
https://www.leiphone.com/news/201710/AixG88EkKRDri7Vd.html  
https://blog.csdn.net/a358463121/article/details/80820878  
https://www.cnblogs.com/initial-h/p/9468974.html  

'''
ECO
'''
attention
'''
crf
'''
词袋模型
'''
ssd
'''
随机森林

  
'''  
dlib、HOG  https://blog.csdn.net/ttransposition/article/details/11874285   
HOG属于特征提取，它统计梯度直方图特征。  
具体来说就是将梯度方向（0->360°）划分为9个区间，将图像化为16x16的若干个block，每个block在化为4个cell（8x8）。  
对每一个cell,算出每一点的梯度方向和模，按梯度方向增加对应bin的值，最终综合N个cell的梯度直方图形成一个高维描述子向量。  
微分算子[1,0,1]，取RGB通道中模值最大的为该像素点的梯度。  
'''  
Gamma校正原理及实现  https://blog.csdn.net/linqianbi/article/details/78617615  
1. 归一化 ：将像素值转换为0-1之间的实数。 算法如下 : ( i + 0. 5)/256  
2. 预补偿 ：像素归一化后的数据以  1 /gamma  为指数的对应值。这一步包含一个 求指数运算。若  gamma  值为  2. 2 ,  则  1 /gamma  为  0. 454545 , 对归一化后的  A  值进行预补偿的结果就 是  0. 783203 ^0. 454545 = 0. 894872 。 
3. 反归一化 ：将经过预补偿的实数值反变换为  0  ～  255  之间的整数值。  
'''  
sift  
DoG尺度空间构造（Scale-space extrema detection）  
关键点搜索与定位（Keypoint localization）  
方向赋值（Orientation assignment）  
关键点描述（Keypoint descriptor）    
'''  
* 图像膨胀类似于“领域扩张”，将图像中的高亮区域或白色部分进行扩张，其运行结果图比原图的高亮区域更大；线条变粗了，主要用于去噪
* 腐蚀类似于“领域被蚕食”，将图像中的高亮区域或白色部分进行缩减细化，其运行结果图比原图的高亮区域更小。
* 开去外白：先腐蚀再膨胀
* 闭去内黑：先膨胀再腐蚀
'''  
* 均值滤波  
均值滤波，是图像处理中最常用的手段，从频率域观点来看均值滤波是一种低通滤波器，高频信号将会去掉，因此可以帮助消除图像尖锐噪声
，实现图像平滑，模糊等功能。理想的均值滤波是用每个像素和它周围像素计算出来的平均值替换图像中每个像素。
* 中值滤波  
中值特别是消除椒盐噪声，中值滤波的效果要比均值滤波更好。中值滤波是跟均值滤波唯一不同是，不是用均值来替换中心每个像素，而是将周围像素和中心像素排序以后，取中值
* 最大最小值滤波
最大最小值滤波是一种比较保守的图像处理手段，与中值滤波类似，首先要排序周围像素和中心像素值，然后将中心像素值与最小和最大像素值比较，如果比最小值小，则替换中心像素为最小值，如果中心像素比最大值大，则替换中心像素为最大值。
* 高斯滤波器
权重只和像素之间的空间距离有关系，无论图像的内容是什么，都有相同的滤波效果。
* 双边滤波器
一种同时考虑了像素空间差异与强度差异的滤波器，因此具有保持图像边缘的特性。
其中 I 是像素的强度值，所以在强度差距大的地方（边缘），权重会减小，滤波效应也就变小。  
总体而言，在像素强度变换不大的区域，双边滤波有类似于高斯滤波的效果，而在图像边缘等强度梯度较大的地方，可以保持梯度。
* 引导滤波
与双边滤波最大的相似之处，就是同样具有保持边缘特性。在引导滤波的定义中，用到了局部线性模型，  
该模型认为，某函数上一点与其邻近部分的点成线性关系，一个复杂的函数就可以用很多局部的线性函数来表。
当需要求该函数上某一点的值时，只需计算所有包含该点的线性函数的值并做平均即可。图像是一个二维函数，而且没法写出解析表达式，因此我们假设该函数的输出与输入在一个二维窗口内满足线性关系，在滤波效果上，引导滤波和双边滤波差不多，在一些细节上，引导滤波较好。引导滤波最大的优势在于，可以写出时间复杂度与窗口大小无关的算法，因此在使用大窗口处理图片时，其效率更高。
