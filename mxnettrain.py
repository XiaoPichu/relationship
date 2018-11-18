# !/usr/bin/python3
# -*- coding:utf-8 -*-
'''
@author = XiaoPichu

faster-r-cnn: each picture contains two boxes

'''
from __future__ import print_function
import os, shutil
import pandas as pd
import csv
import mxnet as mx
import numpy as np
import cv2
import matplotlib.pyplot as plt
import logging
      
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

resultnames = ['ImageID', 'LabelName1', 'LabelName2', 'XMin1', 'XMax1', 'YMin1',\
              'YMax1', 'XMin2', 'XMax2', 'YMin2', 'YMax2', 'RelationshipLabel']

def Network():
    data = mx.symbol.Variable('data')
    conv1 = mx.sym.Convolution(data=data, pad=(1,1), kernel=(3,3), num_filter=24, name="conv1")
    relu1 = mx.sym.Activation(data=conv1, act_type="relu", name= "relu1")
    pool1 = mx.sym.Pooling(data=relu1, pool_type="max", kernel=(2,2), stride=(2,2),name="max_pool1")
    # second conv layer
    conv2 = mx.sym.Convolution(data=pool1, kernel=(3,3), num_filter=48, name="conv2", pad=(1,1))
    relu2 = mx.sym.Activation(data=conv2, act_type="relu", name="relu2")
    pool2 = mx.sym.Pooling(data=relu2, pool_type="max", kernel=(2,2), stride=(2,2),name="max_pool2")

    conv3 = mx.sym.Convolution(data=pool2, kernel=(5,5), num_filter=64, name="conv3")
    relu3 = mx.sym.Activation(data=conv3, act_type="relu", name="relu3")
    pool3 = mx.sym.Pooling(data=relu3, pool_type="max", kernel=(2,2), stride=(2,2),name="max_pool3")
    # first fullc layer
    flatten = mx.sym.Flatten(data=pool3)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500, name="fc1")
    relu3 = mx.sym.Activation(data=fc1, act_type="relu" , name="relu3")
    # second fullc
    fc2 = mx.sym.FullyConnected(data=relu3, num_hidden=43,name="final_fc")
    # softmax loss
    mynet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')       
    mynet.list_arguments()
    
def train():
    batch_size = 64
    X_train_set_as_float = X_train_reshape.astype('float32')
    X_train_set_norm = X_train_set_as_float[:] / 255.0;

    X_validation_set_as_float = X_valid_reshape.astype('float32')
    X_validation_set_norm = X_validation_set_as_float[:] / 255.0 ;


    train_iter =mx.io.NDArrayIter(X_train_set_as_float, y_train_extra, batch_size, shuffle=True)
    val_iter = mx.io.NDArrayIter(X_validation_set_as_float, y_valid, batch_size,shuffle=True)


    print("train set : ", X_train_set_norm.shape)
    print("validation set : ", X_validation_set_norm.shape)


    print("y train set : ", y_train_extra.shape)
    print("y validation set :", y_valid.shape)

    #create adam optimiser
    adam = mx.optimizer.create('adam')

    #checking point (saving the model). Make sure there is folder named models exist
    model_prefix = 'models/chkpt'
    checkpoint = mx.callback.do_checkpoint(model_prefix)
                                           
    #loading the module API. Previously mxnet used feedforward (deprecated)                                       
    model =  mx.mod.Module(
        context = mx.gpu(0),     # use GPU 0 for training if you dont have gpu use mx.cpu()
        symbol = mynet,
        data_names=['data']
       )
                                              
    #actually fit the model for 10 epochs. Can take 5 minutes                                      
    model.fit(
        train_iter,
        eval_data=val_iter, 
        batch_end_callback = mx.callback.Speedometer(batch_size, 64),
        num_epoch = 10, 
        eval_metric='acc',
        optimizer = adam,
        epoch_end_callback=checkpoint
    )
        
    acc = mx.metric.Accuracy()
    model.score(val_iter,acc)
    print(acc)
    
def retrain():    
        #load the model from the checkpoint , we are loading the 10 epoch
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 10)

    # assign the loaded parameters to the module
    mod = mx.mod.Module(symbol=sym, context=mx.cpu())
    mod.bind(for_training=False, data_shapes=[('data', (1,3,32,32))])
    mod.set_params(arg_params, aux_params)
  
def predict():
    img = get_image(url, show=True)
    # compute the predict probabilities
    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    # print the top-5
    prob = np.squeeze(prob)
    prob = np.argsort(prob)[::-1]
    for i in prob[0:5]:
        print('class=%s' %(traffic_labels_dict[i]))
  
if __name__ == '__main__':
    getlabels()