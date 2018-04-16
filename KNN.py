# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from numpy import * 
from operator import * 
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from sklearn.cross_validation import train_test_split
import random
#import matplotlib.pyplot as plt
#import plotly.plotly as py

def knn(input,dataset,labels,k):  
    '至25 皆為計算歐式距離'
    'shape[0] 回傳 行數 shape[1]回傳 列數'
    datasetSize = dataset.shape[0]   
    'tile 複製input 建立(cols,rows)與dataset相同size的複製數據'
    'diffmat = 複製數組 並 與 dataset '
    DiffMat = tile(input,(datasetSize,1)) - dataset  
    '矩陣內所有元素 平方 '
    SqDiffMat = DiffMat ** 2     
    'axis＝0表示按各列相加，axis＝1表示按照各行相加'         
    SqDistances = sum(SqDiffMat,axis = 1)
    'distance 開根號'
    Distances = SqDistances ** 0.5          
    '''''argsort:依照其元素排序 由小至大'''  
    SortDistances = Distances.argsort()  
    LabelsCount = {}  
    '''''K=3 依照 K決定 range並回傳'''  
    for i in range(k):  
        Label = labels[SortDistances[i]] 
        '根據labels統計其相近個數 '
        LabelsCount[Label] = LabelsCount.get(Label,0) + 1
      
    '''''以K個數據相近為主 出現最多即為該label'''  
    MaxNum = 0  
    for key,num in LabelsCount.items():  
        if num > MaxNum:  
            MaxNum = num  
            InputLabel = key  
            
    return InputLabel  
    

def TestData(seed):  
    '用做讀檔練習'
    'names 設定文件標籤 方便後續刪除非必要行列操作 ### skiprows 略過指定資料行讀取'
    iris = pd.read_csv('D:\MSP_LAB (碩一)\圖型辨識_陳洳瑾\iris.data',names=["a","b","c","d","type"])
    '用copy避免動用其原數據'
    iris2 = iris.copy()
    '刪除type列資料'
    iris.pop('type')
    dataset = iris.as_matrix()
    '刪除該dataframe的標籤列'
    iris2.pop('a')
    iris2.pop('b')
    iris2.pop('c')
    iris2.pop('d')
    ### iris2 用作為標籤資料
    '將dataframe 轉為np.array'
    x_train,x_test,y_train,_y_test = train_test_split(dataset,iris2,test_size=0.1,random_state=seed)
#    np_iris2 = iris2.as_matrix()
    '將原本為各行的值 重整回列'
#    labels = np_iris2.reshape(-1)
    #print(labels)
    return x_train,x_test,y_train,_y_test  
'input, input test data 根據skiprows '    


x_train,x_test,y_train,y_test = TestData(300)  
'    '
np_iris2 = y_train.as_matrix()
iris_type = np_iris2.reshape(-1)
np_iris2 = y_test.as_matrix()
iris_type_test = np_iris2.reshape(-1)
'   '
score = 0
'單次執行'
#for i in(range(15)):
##    print(x_test[i], x_train[i] ,iris_type[i])
#    InputLabel = knn(x_test[i], x_train ,iris_type, 3)
#    print(InputLabel,iris_type_test[i])
#    
#    if  InputLabel == iris_type_test[i]:
#        score +=1
#
#print("score",score)

iterative =  100

'執行200次並確認正確率'
'方法不佳 重複給予train data'
accs=0
for c in (range(iterative)):
    x_train,x_test,y_train,y_test = TestData(random.randint(0,65536))
    np_iris2 = y_train.as_matrix()
    iris_type = np_iris2.reshape(-1)
    np_iris2 = y_test.as_matrix()
    iris_type_test = np_iris2.reshape(-1)
    score = 0
    for i in(range(15)):
        InputLabel = knn(x_test[i], x_train ,iris_type, 3)
#        print(InputLabel,iris_type_test[i])
        if  InputLabel == iris_type_test[i]:
            score +=1
#    print("acc",(score/15))
    accs += (score/15)
    
print("正確率",(accs/iterative))

