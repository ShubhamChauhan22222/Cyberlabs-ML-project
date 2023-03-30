#!/usr/bin/env python
# coding: utf-8

# # Importing data

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


# In[2]:


data = pd.read_csv('classification_train.csv')


# In[3]:


data.head(10)


# In[4]:


m=len(data)
m


# In[5]:


#importing dataset
data = pd.read_csv('classification_train.csv')
x = data.iloc[:20000,2:786].values

y = data.iloc[:20000,1].values
y=y[np.newaxis,:] #y = y.reshape(y.shape[0],1)
y=y.T


# In[6]:


m, n = x.shape
x.shape,y.shape 


# # Pre-Processing of data

# In[7]:


mu     = np.mean(x,axis=0)   
sigma  = np.std(x,axis=0) 
x_mean = (x - mu)
x_norm = (x - mu)/sigma
X=x_norm
X.shape


# In[8]:


x1=np.ones((m,1))
x1.shape


# In[9]:


X=np.concatenate((x1,X),axis=1)
X.shape,X


# In[10]:


""" Y: onehot encoded """
unique_values=(np.unique(y))
Y=np.zeros((m,len(unique_values)))
for i in range(m):
    for j in range(len(unique_values)):
        if y[i][0]==unique_values[j]: Y[i][unique_values[j]]=1


# # Cost function

# In[11]:


def sigmoid(z):
   
    g=1/(1+(np.exp(-z)))
      
    return g


# In[12]:


def prediction(X, theta):
    z = np.dot(X, theta)
    return sigmoid(z)


# In[13]:


def cost_function(theta, X, Y):
    m = X.shape[0]
    y_hat = prediction(X, theta)
    return -(1/m) * (np.sum(Y*np.log(y_hat) + (1-Y)*np.log(1-y_hat)))


# In[ ]:





# # Updating Parameters

# In[14]:


def gradient(theta, X, Y):
    m = X.shape[0]
    y_hat = prediction(X, theta)
    return (1/m) * np.dot(X.T, y_hat - Y)


# In[15]:


def gradient_descent(X, Y, max_iter, eta):
   
    cost_list = []
    theta = (np.random.randn(785, 10))*0.01

    for i in range(max_iter):
        theta= theta - eta * gradient(theta, X, Y)
        
#         cost = cost_function(theta, X, Y)
        
        
#         cost_list.append(cost)
        
#         if(i%(max_iter/10) == 0):
#             print("Cost after", i, "iterations is :", cost)
        
        
        
        
    return theta,cost_list


# In[16]:


max_iter=2000
eta=0.01
theta_final,cost_list=gradient_descent(X, Y, max_iter, eta)
theta_final.shape


# In[ ]:





# # Accuracy on test data

# In[17]:


x_test = data.iloc[20000:,2:].values

y_test = data.iloc[20000:,1].values
y_test = y_test.reshape(y_test.shape[0],1)
m_test,n_test=x_test.shape
x_test.shape


# In[18]:


mu     = np.mean(x_test,axis=0)   
sigma  = np.std(x_test,axis=0) 
x_mean = (x_test - mu)
x_norm = (x_test - mu)/sigma
x_test=x_norm
x_test.shape


# In[19]:


x2=np.ones((m_test,1))
x_test=np.concatenate((x2,x_test),axis=1)
x_test.shape


# In[20]:


Z=prediction(x_test, theta_final)
Z.shape


# In[21]:


pred=np.argmax(Z, axis=1)
pred


# In[22]:


pred=pred[np.newaxis,:]
pred=pred.T
pred.shape


# In[23]:


count=0
for i in range(m_test):
    
    if y_test[i,0] ==pred[i,0]:
        count+=1 
print(count)


# In[24]:


accuracy=(count/m_test)*100
print('Accuracy on test data = '+str(math.ceil(accuracy))+'%')
print('Accuracy on test data = '+str(round(accuracy,2))+'%')


# # Finding labels for classification_test

# In[25]:


#importing dataset
data_pred = pd.read_csv('classification_test.csv')
x_pred = data_pred.iloc[:,1:785].values
x_pred.shape
m_pred,n_pred=x_pred.shape


# In[26]:


mu     = np.mean(x_pred,axis=0)   
sigma  = np.std(x_pred,axis=0) 
x_mean = (x_pred - mu)
x_norm = (x_pred - mu)/sigma
x_pred=x_norm
x_pred.shape


# In[27]:


data_pred.shape


# In[28]:


x2=np.ones((m_pred,1))
x2.shape


# In[29]:


x_pred=np.concatenate((x2,x_pred),axis=1)
x_pred.shape


# In[30]:


Z_test=prediction(x_pred, theta_final)
Z_test.shape


# In[31]:


pred_test=np.argmax(Z_test, axis=1)
pred_test


# In[32]:


pred_test=pred_test[np.newaxis,:]
pred_test=pred_test.T
pred_test.shape


# In[33]:


data_pred.insert(0,'label',pred_test)


# In[34]:


#saving the dataframe as a csv file
data_pred.to_csv('classification_test_pred.csv',index=False)


# In[ ]:





# In[ ]:




