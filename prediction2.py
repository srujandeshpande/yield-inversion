# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 16:27:33 2019

@author: Srujan Deshpande
"""

# -*- coding: utf-8 -*-
"""
By: Srujan Vasudevrao Deshpande PES2201800105
    Vaibhav Gupta PES2201800093
    CSE Section B PES University Electronic City Campus
"""

# =============================================================================
# Imports
# =============================================================================
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# =============================================================================
# Reading and formatting data
# =============================================================================
df10 = pd.read_csv('DGS10.csv')
df3 = pd.read_csv('DTB3.csv')
df5 = pd.read_csv('DGS5.csv')
df3y = pd.read_csv('DGS3.csv')
dfrec = pd.read_csv('JHDUSRGDPBR.csv')

df10['DATE']= pd.to_datetime(df10['DATE'])
df10 = df10[df10.DGS10 != '.']
df10['DGS10'] = df10['DGS10'].astype(float)
newdf10 = df10[(df10['DATE'].dt.year >= 1968)]

df3['DATE']= pd.to_datetime(df3['DATE'])
df3 = df3[df3.DTB3 != '.']
df3['DTB3'] = df3['DTB3'].astype(float)
newdf3 = df3[(df3['DATE'].dt.year >= 1968)]

df5['DATE']= pd.to_datetime(df5['DATE'])
df5 = df5[df5.DGS5 != '.']
df5['DGS5'] = df5['DGS5'].astype(float)
newdf5 = df5[(df5['DATE'].dt.year >= 1968)]

df3y['DATE']= pd.to_datetime(df3y['DATE'])
df3y = df3y[df3y.DGS3 != '.']
df3y['DGS3'] = df3y['DGS3'].astype(float)
newdf3y = df3y[(df3y['DATE'].dt.year >= 1968)]

dfrec['DATE']= pd.to_datetime(dfrec['DATE'])
dfrec['JHDUSRGDPBR'] = dfrec['JHDUSRGDPBR'].astype(bool)
newdfrec = dfrec[(dfrec['DATE'].dt.year >= 1968)]

# =============================================================================
# Merging the dataframes
# =============================================================================
newdf = pd.merge(newdf10,newdf3, on="DATE")
newdf = pd.merge(newdf,newdf5,on="DATE")
newdf = pd.merge(newdf,newdf3y,on="DATE")
newdf = newdf.dropna()
newdf = newdf[newdf.DGS10 != '.']
newdf = newdf[newdf.DTB3 != '.']
newdf = newdf[newdf.DGS5 != '.']
newdf = newdf[newdf.DGS3 != '.']

mergednew = pd.merge_asof(newdf, newdfrec, on="DATE")
mergednew.dropna()

# =============================================================================
# Splitting into test and train and scaling
# =============================================================================
X = mergednew.iloc[:,[1,2,3,4]].values
y = mergednew.iloc[:,5].values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)
x2_train, x2_test, y2_train, y2_test = train_test_split(X, y, test_size = 0.005, random_state = 0, shuffle = False)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# =============================================================================
# Lightgbm Model
# =============================================================================
d_train = lgb.Dataset(x_train, y_train)
params = {}
params['learning_rate'] = 0.3
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 100
params['min_data'] = 50
params['max_depth'] = 10
clf = lgb.train(params, d_train, 100)

y_pred=clf.predict(x_test)
# =============================================================================
# Converting Data to Binary
# =============================================================================
for i in range(0,len(y_pred)):
    if y_pred[i]>=0.245:
        y_pred[i]=1
    else:
        y_pred[i]=0

# =============================================================================
# Accuracy Testing
# =============================================================================
#Confusion matrix
cm = confusion_matrix(y_test, y_pred)
#Accuracy
accuracy=accuracy_score(y_pred,y_test)

print("Confusion Matrix:\n",cm)
print("Accuracy: ",accuracy)

# =============================================================================
# Checking the last 65 days
# =============================================================================
x2_train, x2_test, y2_train, y2_test = train_test_split(X, y, test_size = 0.005, random_state = 0, shuffle = False)
newy = clf.predict(x2_test)
x2_test
for i in range(0,len(newy)):
    if newy[i]>=0.245:
        print(len(newy)-i,newy[i])
        newy[i]=1
    else:
        newy[i]=0

len(newy)
