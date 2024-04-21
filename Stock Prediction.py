#!/usr/bin/env python
# coding: utf-8

# In[51]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
import tensorflow as tf
from tensorflow import keras


# In[52]:


import pandas as pd


# In[53]:


stock=pd.read_csv("C:\\Users\\Super\\Downloads\\NFLX.csv",index_col='Date',parse_dates=True)


# In[54]:


stock.head()


# In[55]:


stock.describe()


# In[56]:


stock.info()


# In[57]:


stock.head()


# There is no NaN data
# Date handled already
# next step is ploting our Data to see what is the important feature

# In[58]:


for columns in stock.columns:
    plt.figure(figsize=(12,4))
    plt.title(f"Stock {columns} Price")
    plt.plot(stock.index,stock[columns])
    plt.xticks(rotation=45)


# In[59]:


plt.figure(figsize=(12,4))
plt.title("Stock Price")
for columns in stock.columns:
    if(columns !='Volume'):
        plt.plot(stock.index,stock[columns],label=columns)
plt.xticks(rotation=45)
plt.legend()


# # Data Normalization

# In[60]:


stock=stock['Close']
stock.shape


# In[61]:


stock


# In[62]:


scaler=MinMaxScaler(feature_range=(0,1))


# In[63]:


# df=scaler.fit_transform(np.array(data['Close']).reshape(-1,1))
df=scaler.fit_transform(np.array(stock).reshape([stock.shape[0],1]))


# In[64]:


def create_seq(stock,time_step=60):
    X=[]
    y=[]
    for i in range(len(stock)-time_step-1):
        X.append(stock[i:(i+time_step)])
        y.append(stock[i+time_step])
    return X,y


# In[65]:


time_step=100
X,y=create_seq(df,time_step)
len(X),len(y)


# # reshape input to be [samples, time-steps, features] which is required for LSTM

# In[66]:


X=np.array(X)
X=X.reshape(X.shape[0],X.shape[1],1)
y=np.array(y)
X.shape,y.shape


# Splitting the data
# 

# In[67]:


X_train,X_test,y_train,y_test=X[:int(stock.shape[0]*0.8)],X[int(stock.shape[0]*0.8):],y[:int(stock.shape[0]*0.8)],y[int(stock.shape[0]*0.8):]


# In[68]:


X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[69]:


model=Sequential()
model.add(LSTM(128,return_sequences=True,input_shape=X_train[0].shape))
model.add(LSTM(64,return_sequences=True))
model.add(LSTM(32))
model.add(Dense(16,activation='relu'))
model.add(Dense(1))
model.compile(optimizer= keras.optimizers.Adam(learning_rate=0.001),loss="mean_squared_error",metrics=[keras.metrics.RootMeanSquaredError()])
model.summary()
callback=tf.keras.callbacks.EarlyStopping(monitor='loss',patience=5,min_delta=0.1)


# In[50]:


model.fit(X_train,y_train,epochs=100,callbacks=[callback])


# In[ ]:





# In[ ]:




