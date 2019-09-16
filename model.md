#-*- coding=ytf-8 -*-  
'''  
semantic module  
'at','on','inside_of','under',\ 'wears','interacts_with','hits','holds','plays',  
数据库中主要由这3种关系组成: geometric(几何): 50.9%, possessive(所有格): 40.9% semantic(语义): 8.7%.  
衣服、身体部件大多是所有格关系；家具、建筑大多是几何关系；人大多是语义关系的主语。  
'''  
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
   
'''  
Spatial Module    
宾语对主语 主语对联合 宾语对联合    
  
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
  
'''  
Visual Module  

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
'''
sift
'''
'''
attention
'''
'''
crf
'''
'''
dlib
'''
'''
hog
'''
'''
词袋模型
'''
'''
ssd
'''
'''  
fasterrcnn  
![框图](https://github.com/XiaoPichu/relationship/blob/master/fasterrcnn.png)  
'''  
'''
featnms
'''
'''
随机森林
'''
'''
中值滤波
'''
