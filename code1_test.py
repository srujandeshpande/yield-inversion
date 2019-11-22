# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 04:39:30 2019

@author: Srujan Deshpande
"""
import pandas as pd
#import pandas_datareader as pdr
import datetime
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



df10 = pd.read_csv('DGS10.csv')
df3 = pd.read_csv('DTB3.csv')
dfrec = pd.read_csv('JHDUSRGDPBR.csv')

df10['DATE']= pd.to_datetime(df10['DATE'])
df10 = df10[df10.DGS10 != '.']
df10['DGS10'] = df10['DGS10'].astype(float)
newdf10 = df10[(df10['DATE'].dt.year >= 1968)]

df3['DATE']= pd.to_datetime(df3['DATE'])
df3 = df3[df3.DTB3 != '.']
df3['DTB3'] = df3['DTB3'].astype(float)
newdf3 = df3[(df3['DATE'].dt.year >= 1968)]

dfrec['DATE']= pd.to_datetime(dfrec['DATE'])
dfrec['JHDUSRGDPBR'] = dfrec['JHDUSRGDPBR'].astype(bool)
newdfrec = dfrec[(dfrec['DATE'].dt.year >= 1968)]

newdf = pd.merge(newdf10,newdf3, on="DATE")
#newdf2 = pd.merge(newdf,dfrec, on="DATE")
#newdf2 = newdf2.dropna()
newdf = newdf.dropna()
newdf = newdf[newdf.DGS10 != '.']
newdf = newdf[newdf.DTB3 != '.']

newdf

#newdf['diff'] = newdf.DGS10 - newdf.DTB3 

newdf

mergednew = pd.merge_asof(newdf, newdfrec, on="DATE")
mergednew.dropna()

#x_train, x_test, y_train, y_test = train_test_split(mergednew, mergednew, test_size = 0.25, random_state = 0)

X = mergednew.iloc[:,[1,2]].values
y = mergednew.iloc[:,4].values

#x_train, x_test, y_train, y_test = train_test_split(mergednew['diff'], mergednew['JHDUSRGDPBR'], test_size = 0.25, random_state = 0)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

len(x_train)
len(y_train)
len(x_test)
len(y_test)

d_train = lgb.Dataset(x_train, y_train)
params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 10
params['min_data'] = 50
params['max_depth'] = 10
clf = lgb.train(params, )

accuracy
cm

len(y_pred)
clf = lgb.train(params, d_train, 100)

#Prediction
y_pred=clf.predict(x_test)
#convert into binary values
for i in range(0,len(y_pred)):
    print (y_pred[i])
    if y_pred[i]>=.2:       # setting threshold to .5
       y_pred[i]=1
    else:  
       y_pred[i]=0


#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#Accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred,y_test)
y_pred[0]
y_test

