# -*- coding: utf-8 -*-
"""
By: Srujan Vasudevrao Deshpande PES2201800105
    Vaibhav Gupta PES2201800093
    CSE Section B PES University Electronic City Campus
"""

import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# Reading and formatting data
# =============================================================================
df= pd.read_csv('yeildcurve.csv')
df['Date']= pd.to_datetime(df['Date'])

fullrecdf= pd.read_csv('us.csv')
fullrecdf['DATE']= pd.to_datetime(fullrecdf['DATE'])
fullrecdf['JHDUSRGDPBR']= fullrecdf['JHDUSRGDPBR'].astype(bool)

# =============================================================================
# GRAPH 1: Normal Yield Curve 6/13/2013 13 June 2013
# =============================================================================
values = [0.04,0.05,0.08,0.14,0.32,0.55,1.11,1.6,2.19,2.99,3.33]
names = ['1 Mo','3 Mo','6 Mo','1 Yr','2 Yr','3 Yr','5 Yr','7 Yr','10 Yr','20 Yr','30 Yr']
plt.xlabel('Bond Duration')
plt.ylabel('Interest Rate (%)')
plt.title('Interest Rates on 13 June 2013')
plt.plot(names,values)
plt.scatter(names,values)
plt.show()

# =============================================================================
# GRAPH 2: Inverted Yield 1/23/2007 23 January 2007
# =============================================================================
values = [5.02,5.14,5.18,5.1,4.94,4.85,4.81,4.81,4.81,5.0,4.9]
names = ['1 Mo','3 Mo','6 Mo','1 Yr','2 Yr','3 Yr','5 Yr','7 Yr','10 Yr','20 Yr','30 Yr']
plt.xlabel('Bond Duration')
plt.ylabel('Interest Rate (%)')
plt.title('Interest Rates on 23 January 2007')
plt.plot(names,values)
plt.scatter(names,values)
plt.show()

# =============================================================================
# GRAPH 3: Recession from 1990 onwards
# =============================================================================
recdf = fullrecdf[(fullrecdf['DATE'].dt.year >= 1990)]
plt.plot(recdf['DATE'],recdf['JHDUSRGDPBR'],color="red")
plt.xlabel('Date')
plt.ylabel('Recession')
plt.title('Recessions 1990 onwards')
plt.show()


# =============================================================================
# GRAPH 4: 2 Year Interest rate vs time
# =============================================================================
y=df['2 Yr']
x=df['Date']
plt.xlabel('Date')
plt.ylabel('Interest Rate (%)')
plt.plot(x,y)
plt.title("2 Year Treasury Bond vs. Time")
plt.show()

# =============================================================================
# GRAPH 5: 10-2 Year Interest Rate vs time
# =============================================================================
newdf = df[['Date','2 Yr','10 Yr']]
newdf['sub'] = newdf['10 Yr']- newdf['2 Yr']
newdf.info()
y=newdf['sub']
x=newdf['Date']
plt.xlabel('Date')
plt.ylabel('Difference in Interest Rate (%)')
plt.title("10 Year Yield - 2 Year Yield vs. Time")
plt.plot(x,y)
plt.axhline(y=0.0, color='black', linestyle='-')
plt.show()

# =============================================================================
# GRAPH 6: 10-2 Year Interest Rate vs time with Recession Shading
# =============================================================================
newdf = df[['Date','2 Yr','10 Yr']]
newdf['sub'] = newdf['10 Yr']- newdf['2 Yr']
newdf.info()
y=newdf['sub']
x=newdf['Date']
recdf = fullrecdf[(fullrecdf['DATE'].dt.year >= 1990)]
plt.xlabel('Date')
plt.ylabel('Diference in Interest Rate (%)')
plt.title("10 Year Yield - 2 Year Yield vs. Time with Recessions Highlighted")
plt.plot(x,y)
plt.plot(recdf['DATE'],recdf['JHDUSRGDPBR'],color="red")
plt.axhline(y=0.0, color='black', linestyle='-')
plt.show()

