# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 04:41:54 2019

@author: Srujan Deshpande
"""

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
