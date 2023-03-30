#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import math

import random


# # Importing train_data

# In[2]:


#importing dataset
data = pd.read_csv('classification_train.csv')


# train = pd.read_csv('classification_train.csv',
#                    skiprows = lambda i: random.random() > 0.50)
# x = train.iloc[:,2:786].values
# 
# 
# y = train.iloc[:,1].values
# 

# In[3]:


x = data.iloc[:20000,2:].values

y = data.iloc[:20000,1].values
y=y[np.newaxis,:] #y = y.reshape(y.shape[0],1)
y=y.T


# In[4]:


m,n=x.shape
m,n


# In[5]:


k=60


# # DEFINE x2 as test data

# In[6]:


#x2=data_pred.iloc[:,1:].values
x2 = data.iloc[20000:,2:].values
M,N=x2.shape
M,N


# distance_j = np.sum((x-x2[3,:])**2,axis=0)
# distance_j.shape

# # Step1 distance

# In[7]:


def Euclidean_distance():
    distance=np.zeros((m,M))    
    for j in range(M):
        
        distance[:,j]=np.sqrt(np.sum((x-x2[j,:])**2,axis=1))
        
    return distance
Euclidean_distance=Euclidean_distance()


# In[8]:


Euclidean_distance.shape


# # Sorting and finding labels

# In[9]:


KNN_PRED_FINAL=[]

for j in range(M):
    c=(Euclidean_distance[:,j]).reshape(m,1)
    distance_label=np.concatenate((c,y.reshape(m,1)), axis=1)
    k_sorted=(distance_label[distance_label[:, 0].argsort()])[0:k,:]
    
    b=(np.unique(y))
    count=np.zeros((len(b),1))
    for i in range(k):
        for j in range(len(b)):
            if k_sorted[i,1]==b[j]:    count[b[j],0]+=1
                
#     pred_test=np.argmax(k_sorted)
    KNN_PRED_FINAL.append(np.argmax(count))


# In[ ]:





# In[ ]:





# # Accuracy of test data

# In[10]:


y_test = data.iloc[20000:,1].values


# In[11]:


count_pred=0
for i in range(M):
    
    if y_test[i] == KNN_PRED_FINAL[i]:
        count_pred+=1 
print(count_pred)


# In[12]:


accuracy=(count_pred/M)*100
print('Accuracy on test data = '+str(accuracy)+'%')


# In[ ]:





# In[ ]:




