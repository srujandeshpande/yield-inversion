# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 04:41:54 2019

@author: Srujan Deshpande
"""

import numpy as np
import matplotlib.pyplot as plt
df= pd.read_csv('us.csv', delimiter=',' , nrows= 208 , skiprows=[1])
df
df['DATE']= pd.to_datetime(df['DATE'])
df.info()
y_pos = np.arange(len(df['DATE']))
plt.bar(y_pos,df['JHDUSRGDPBR'])
plt.xticks(y_pos,df['DATE'])
plt.xlabel('year')
plt.ylabel('rec')
plt.show()

import matplotlib.pyplot as plt
import pandas as pd
df= pd.read_csv('yeildcurve.csv', delimiter=',' , nrows= 7477 , skiprows=[1])
df
df['Date']= pd.to_datetime(df['Date'])
df.info()
y=df['3 Mo']
x=df['Date']
plt.xlabel('year')
plt.ylabel('interest rate(%)')
plt.plot(x,y)
plt.show()
print(df['Date'])



import matplotlib.pyplot as plt
import pandas as pd
df= pd.read_csv('yeildcurve.csv', delimiter=',' , nrows= 7477)
df['Date']= pd.to_datetime(df['Date'])
df.info()
newdf = df[['Date','2 Yr','10 Yr']]
newdf.info()
sub= newdf['10 Yr']- newdf['2 Yr']
newdf['sub'] = newdf['10 Yr']- newdf['2 Yr']
newdf.info()

y=newdf['sub']
x=newdf['Date']
plt.xlabel('year')
plt.ylabel('interest rate(%)')
plt.plot(x,y)
plt.axhline(y=0.0, color='r', linestyle='-')
plt.show()