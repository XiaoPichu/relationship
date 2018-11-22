# !/usr/bin/python3
# -*- coding:utf-8 -*-
'''
@author = XiaoPichu

'''
from __future__ import print_function
from random import shuffle

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
