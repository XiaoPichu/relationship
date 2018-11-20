# !/usr/bin/python3
# -*- coding:utf-8 -*-

from __future__ import print_function
import os, shutil
import pandas as pd
import csv

workspace = os.getcwd()
dataspace = '..\\data\\'
labelscsv = dataspace+'labels.csv'
attributioncsv1 = dataspace+'labelname1.csv'
attributioncsv2 = dataspace+'labelname2.csv'
attributioncsv3 = dataspace+'isname.csv'
attribution = ['at','on','holds','plays','interacts_with',\
               'wears','is','inside_of','under','hits']
isatt = ['Transparent','Plastic','made of Textile','made of Leather','Wooden']
classnames = ['ImageID', 'LabelName1', 'LabelName2', 'XMin1', 'XMax1', 'YMin1',\
              'YMax1', 'XMin2', 'XMax2', 'YMin2', 'YMax2', 'RelationshipLabel']
n_class = len(classnames)
 
def replacelabel(csvname):
    dflabel = pd.read_csv(csvname,header = None,names = classnames) 
    dfattribution_1 = open(attributioncsv1,'r')
    dfattribution_2 = open(attributioncsv2,'r')
    dfattribution_3 = open(attributioncsv3,'r')
    dfattribution1 = [row for row in csv.reader(dfattribution_1)]
    dfattribution2 = [row for row in csv.reader(dfattribution_2)]
    dfattribution3 = [row for row in csv.reader(dfattribution_3)]
    for i in range(len(dfattribution1)):
        dflabel.loc[dflabel.loc[:,'LabelName1']==dfattribution1[i][0],'LabelName1'] = dfattribution1[i][1]
        dflabel.loc[dflabel.loc[:,'LabelName2']==dfattribution1[i][0],'LabelName2'] = dfattribution1[i][1]
    for i in range(len(dfattribution2)):
        dflabel.loc[dflabel.loc[:,'LabelName1']==dfattribution2[i][0],'LabelName1'] = dfattribution2[i][1]
        dflabel.loc[dflabel.loc[:,'LabelName2']==dfattribution2[i][0],'LabelName2'] = dfattribution2[i][1]
    for i in range(len(dfattribution3)):
        dflabel.loc[dflabel.loc[:,'LabelName2']==dfattribution3[i][0],'LabelName2'] = dfattribution3[i][1]
    dflabel.to_csv(csvname,encoding='utf-8',index=False,header = None)  
    dfattribution_1.close()
    dfattribution_2.close()  
    dfattribution_3.close()  
    
tmpiscsv = 'is.csv'
tmpnoiscsv = 'nois.csv'
def splitlabel():
    df = pd.read_csv(labelscsv,header = None,names = classnames)   
    df[df.loc[:,'RelationshipLabel']=='is'].to_csv(tmpiscsv,encoding='utf-8',index=False)
    df[df.loc[:,'RelationshipLabel']!='is'].to_csv(tmpnoiscsv, header = None,encoding='utf-8',index=False)
   
tmpnoisdic = 'noisdictionary.csv'
tmpisdic = 'isdictionary.csv'
def setdictionarycsv(csvnamein,csvnameout):
    df = pd.read_csv(csvnamein,header = None,names = classnames)
    df = df.drop_duplicates(subset=['LabelName1','LabelName2','RelationshipLabel'],keep='first')
    df[['LabelName1','LabelName2','RelationshipLabel']].to_csv(csvnameout,encoding='utf-8',index=False,header = None)

tmpnoislabel = 'setnoislabel.txt'
def getnolabels(csvnamein,csvnameout):
    with open(csvnamein,'r') as _:
        allobjects = [row[:2] for row in csv.reader(_)[1:]]
        out = []
        for allobject in allobjects:
            out.append(allobject[0])
            out.append(allobject[1])
        out = sorted(list(set(out)))
        with open(csvnameout,'w') as outwriter:
            outwriter.write('\n'.join(out))

tmpislabel = 'setislabel.txt'            
def getlabels(csvnamein,csvnameout):
    with open(csvnamein,'r') as _:
        allobjects = [row[0] for row in csv.reader(_)[1:]]
        out = sorted(list(set(allobjects)))
        with open(csvnameout,'w') as outwriter:
            outwriter.write('\n'.join(out))
if __name__ == '__main__':
    #replacelabel(labelscsv)
    #splitlabel()
    #setdictionarycsv(tmpnoiscsv,tmpnoisdic)
    getnolabels(tmpnoisdic,tmpnoislabel)
    getlabels(tmpisdic,tmpislabel)