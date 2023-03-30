#!/usr/bin/env python
# coding: utf-8

# # Importing Python libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math 


# # Making numpy arrays x and y

# In[2]:


#importing dataset
data = pd.read_csv('linear_train.csv')
x = data.iloc[:,1:-1].values

y = data.iloc[:,-1].values
y=y[np.newaxis,:]
y=y.T
y.shape


# In[3]:


mu     = np.mean(x,axis=0)   
sigma  = np.std(x,axis=0) 
x_mean = (x - mu)
x_norm = (x - mu)/sigma


# # Loading data

# In[4]:


x_norm


# data = pd.read_csv('linear_train.csv')

# In[5]:


data.head(10)


# In[6]:


m=len(data)
m


# In[7]:


m,n=x.shape


# In[ ]:





# # Cost Function

# In[ ]:





# In[8]:


def compute_cost(x_norm, y, w, b): 
                               
    f_wb = np.dot(x_norm, w) + b           
    cost = (f_wb - y) 
    cost = cost**2
    cost = cost / (2 * m)
    cost=np.sum(cost)
    
    return cost


# In[ ]:





# # Gradient Descent

# In[9]:


def compute_gradient(x_norm, y, w, b): 
   
    m,n = x_norm.shape           #(number of examples, number of features)
    y_hat = np.dot(x_norm,w) + b                      
    err = (y_hat - y)
    
    # dj_db
    dj_db = np.sum(err)
    dj_db = dj_db / m 
    
    #dj_dw
    
    dj_dw = (1/m)*(np.dot(x_norm.T, err))   
#     dj_dw+=(lambda_/m)*(np.sum(w))                               
        
    return dj_db, dj_dw


# In[10]:


def gradient_descent(x_norm, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    J_history = []
    w_history = []
    
    w=w_in
    b = b_in
    
    for i in range(num_iters):

        
        dj_db,dj_dw = gradient_function(x_norm, y, w, b)   
            
        w = w - alpha * dj_dw
                    
        b = b - alpha * dj_db  
         # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  cost_function(x_norm, y, w, b)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w, b, J_history, w_history #return w and J,w history for graphing 


# In[ ]:





# In[11]:


initial_w = np.zeros((n,1))
initial_b = 0
# some gradient descent settings
iterations = 1000
alpha = 0.01
# run gradient descent 
w_norm,b_norm,J_history, w_history = gradient_descent(x_norm ,y, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations)
print("w,b found by gradient descent:", w_norm, b_norm)


# In[12]:


compute_cost(x_norm, y, w=w_norm, b=b_norm)


# In[13]:


t = np.arange(0, iterations)
plt.plot(t,J_history )
plt.xlabel("iterations")
plt.ylabel("cost")
plt.title("iterations vs cost")

plt.show()


# In[ ]:





# # RMSE and R2 score

# In[ ]:





# In[14]:


# Model Evaluation - RMSE
def rmse(Y, Y_pred):
    rmse = np.sqrt(sum((Y - Y_pred) ** 2) / (m))
    return rmse

# Model Evaluation - R2 Score
def r2_score(Y, Y_pred):
    mean_y = np.mean(Y)
    ss_tot = sum((Y - mean_y) ** 2)
    ss_res = sum((Y - Y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

Y_pred = np.dot(x_norm, w_norm) + b_norm

print("RMSE")
print(rmse(y, Y_pred))
print("R2 Score")
print(r2_score(y, Y_pred))


# In[ ]:





# In[ ]:





# # Predicting labels and saving csv file

# In[ ]:





# In[15]:


#importing dataset
data_pred = pd.read_csv('linear_test_data.csv')
x_test= data_pred.iloc[:,1:21].values


# In[16]:


mu     = np.mean(x_test,axis=0)   
sigma  = np.std(x_test,axis=0) 
x_test_norm = (x_test - mu)/sigma
x_test_norm.shape


# In[17]:


y_pred=np.dot(x_test_norm, w_norm) + b_norm


# In[18]:


y_pred.shape


# In[19]:


data_pred.insert(0,'label',y_pred)


# In[20]:


#saving the dataframe as a csv file
data_pred.to_csv('linear_test_data_pred.csv',index=False)


# In[ ]:




