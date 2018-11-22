# !/usr/bin/python3
# -*- coding:utf-8 -*-
'''
@author = XiaoPichu

'''
from __future__ import print_function
from random import shuffle
import logging

# 第一步，创建一个logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Log等级总开关
# 第二步，创建一个handler，用于写入日志文件
logfile = 'logging.log'
fh = logging.FileHandler(logfile, mode='w')
fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
# 第三步，定义handler的输出格式
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(formatter)
# 第四步，添加
logger.addHandler(fh)
# 日志
logger.debug('this is a logger debug message')
logger.info('this is a logger info message')
logger.warning('this is a logger warning message')
logger.error('this is a logger error message')
logger.critical('this is a logger critical message')


def splitdataset(ratio):
    with open('nois.csv','r') as df:
        df = df.readlines()[1:]
        shuffle(df)
        train = df[:int(len(df)*ratio)]
        valid = df[int(len(df)*ratio):]
        with open('train.csv','w') as trainwriter:
            trainwriter.write(''.join(train))
        with open('valid.csv','w') as validwriter:
            validwriter.write(''.join(valid))

def valid(flag = 0):
    with open('noisdictionary.csv','r') as dic:
        dic = dic.readlines()[1:]
        dict = []
        for i in dic:
            line = i.strip('\n').split(',')
            dict.append([line[0],line[1],line[2]])
        
        precise = []
        with open('valid.csv','r') as validread:
            validread = validread.readlines()
            for i in validread:
                line = i.strip('\n').split(',')
                #print([dic[2] for dic in dict if dic[0]==line[1] and dic[1]==line[2]])
                #print(line[11])  
                precise.append(line[11] in [dic[2] for dic in dict if dic[0]==line[1] and dic[1]==line[2]])
        print(sum(precise)*1.0/len(precise))


#def train():
    
#def getrelation():
    
        

if __name__ == '__main__':
    #splitdataset(0.7)
    valid()
