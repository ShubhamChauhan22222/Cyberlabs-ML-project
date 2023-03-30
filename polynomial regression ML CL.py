#!/usr/bin/env python
# coding: utf-8

# # Importing data

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


# In[2]:


a=np.zeros((2,3))


# In[ ]:





# In[3]:


data = pd.read_csv('polynomial_train.csv')


# In[4]:


data.head(10)


# In[5]:


m=len(data)
m


# In[6]:


data.columns


# In[7]:


#importing dataset
x = data.iloc[:30000,1:-1].values

y = data.iloc[:30000,-1].values
y=y[np.newaxis,:]
y=y.T
y.shape
A=x[:,0]
B=x[:,1]
C=x[:,2]


# In[8]:


m,n=x.shape
m,n
A.shape,y.shape


# # Plots

# In[9]:


plt.scatter(A,y,marker='x',c='r')
plt.xlabel("A")
plt.ylabel("label")
plt.title("A vs label")
plt.show()

plt.scatter(B,y,marker='x',c='r')
plt.xlabel("B")
plt.ylabel("label")
plt.title("B vs label")
plt.show()

plt.scatter(C,y,marker='x',c='r')
plt.xlabel("C")
plt.ylabel("label")
plt.title("C vs label")
plt.show


# In[10]:


mu     = np.mean(x,axis=0)   
sigma  = np.std(x,axis=0) 
x_mean = (x - mu)
x_norm = (x - mu)/sigma


# In[11]:


plt.scatter(x_norm[:,0],y,marker='x',c='r')
plt.xlabel("A_norm")
plt.ylabel("label")
plt.title("A_norm vs label")
plt.show()

plt.scatter(x_norm[:,1],y,marker='x',c='r')
plt.xlabel("B_norm")
plt.ylabel("label")
plt.title("B_norm vs label")
plt.show()

plt.scatter(x_norm[:,2],y,marker='x',c='r')
plt.xlabel("C_norm")
plt.ylabel("label")
plt.title("C_norm vs label")
plt.show


# # n- degree polynomial

# In[12]:


def poly(x,pow):
  m=x.shape[0]
  n=(((pow)*(pow+1)*((2*pow)+1))+(9*(pow)*(pow+1))+(12*pow))//12
  x_poly=np.zeros((m,n))
  k=0
  i=0
  j=0
  while(pow!=0):
    x_poly[:,k]=((x[:,0]**(i))*(x[:,1]**(j))*(x[:,2]**(pow-i-j)))
    j=j+1
    k=k+1
    if(j>(pow-i)):
      i=i+1
      j=0
    if(i>pow):
      pow=pow-1
      i=0
  return x_poly


# # Cost function

# In[13]:


def predict(i, w, b): 
    xi = X[i]
    xi=xi[np.newaxis,:]
    #xi=xi.T
    p = np.dot(xi, w) + b     
    return p


# In[14]:


def compute_cost_reg(X, y, w, b,lambda_=1): 
                               
    f_wb = np.dot(X, w) + b           
    cost = (f_wb - y) 
    cost = cost**2
    cost = cost / (2 * m)
    cost=np.sum(cost)
    reg1 = (lambda_/(2*m))*(np.sum(w**2)) 
    reg_cost = cost + reg1
    return reg_cost


# In[ ]:





# # Gradient Descent

# In[15]:


def compute_gradient_reg(X, y, w, b,lambda_=1): 
   
    m,n = X.shape           #(number of examples, number of features)
    
    y_hat = np.dot(X,w) + b                      
    err = (y_hat - y)
    
    # dj_db
    dj_db = np.sum(err)
    dj_db = dj_db / m 
    
    #dj_dw
    
    dj_dw = (1/m)*(np.dot(X.T, err))  
    dj_dw+=(lambda_/m)*(np.sum(w))
                                   
        
    return dj_db, dj_dw


# In[16]:


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    J_history = []
    w_history = []
   
    w = w_in
    b = b_in
    
    for i in range(num_iters):

        
        dj_db,dj_dw = gradient_function(X, y, w, b)   
            
        w = w - alpha * dj_dw
                    
        b = b - alpha * dj_db    
        
        
         # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  cost_function(X, y, w, b)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w, b, J_history, w_history #return w and J,w history for graphing 


# # Degree selection

# In[ ]:





# In[17]:


x_cv = data.iloc[30000:,1:-1].values
y_cv = data.iloc[30000:,-1].values
y_cv=y_cv[np.newaxis,:]
y_cv=y_cv.T


# In[18]:


mu     = np.mean(x_cv,axis=0)   
sigma  = np.std(x_cv,axis=0) 
x_mean = (x_cv - mu)
x_cv = (x_cv - mu)/sigma


# In[19]:


cost_train=[]
cost_cv=[]
for i in range(5):


    X=poly(x_norm,i)
    m,n=X.shape
    
    iterations = 1500
    alpha = 0.01
    
    
    w = np.zeros((n,1))
    b = 0
    
    for j in range(iterations):

        
        dj_db,dj_dw = compute_gradient_reg(X, y, w, b)   
            
        w = w - alpha * dj_dw
                    
        b = b - alpha * dj_db  
      
    cost_train_i=compute_cost_reg(X, y, w, b,lambda_=1)
    
    cost_train.append(cost_train_i)
    
    
    
    X_cv=poly(x_cv,i)
    m_cv,n_cv=X_cv.shape
    
    iterations = 1500
    alpha = 0.01
    
    
    w_cv = np.zeros((n_cv,1))
    b_cv = 0
    
    for j in range(iterations):

        
        dj_db,dj_dw = compute_gradient_reg(X_cv,y_cv , w_cv, b_cv)   
            
        w_cv = w_cv - alpha * dj_dw
                    
        b_cv = b_cv - alpha * dj_db  
      
    cost_cv_i=compute_cost_reg(X_cv, y_cv, w_cv, b_cv,lambda_=1)
    
    cost_cv.append(cost_cv_i)
    

t = np.arange(0, 5)
plt.plot(t,cost_train, color='r',label='cost_train')
plt.plot(t,cost_train,color='b',label='cost_cv')

plt.xlabel("degree")
plt.ylabel("cost")
plt.title("degree vs cost")

plt.show()
    


# # Updating Parameters

# In[20]:


X=poly(x_norm,5)
m,n=X.shape


# In[ ]:





# In[21]:


initial_w = np.zeros((n,1))
initial_b = 0
# some gradient descent settings
iterations = 20000
alpha = 0.001
# run gradient descent 
w_norm,b_norm,J_history, w_history = gradient_descent(X ,y, initial_w, initial_b, compute_cost_reg, compute_gradient_reg, alpha, iterations)
print("w,b found by gradient descent:", w_norm, b_norm)


# In[22]:


compute_cost_reg(X, y, w= w_norm, b= b_norm)


# In[23]:


t = np.arange(0, iterations)
plt.plot(t,J_history)
plt.xlabel("iterations")
plt.ylabel("cost")
plt.title("iterations vs cost")

plt.show()


# In[ ]:





# # RMSE and R2 of train data

# In[24]:


# Model Evaluation - RMSE
def rmse(Y, Y_pred):
    rmse = np.sqrt(sum((Y - Y_pred) ** 2) / m)
    return rmse

# Model Evaluation - R2 Score
def r2_score(Y, Y_pred):
    mean_y = np.mean(Y)
    ss_tot = sum((Y - mean_y) ** 2)
    ss_res = sum((Y - Y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

Y_pred = np.dot(X, w_norm) + b_norm

print("RMSE")
print(rmse(y, Y_pred))
print("R2 Score")
print(r2_score(y, Y_pred))


# In[ ]:





# # RMSE and R2 of Cross validation set

# In[25]:


mu     = np.mean(x_cv,axis=0)   
sigma  = np.std(x_cv,axis=0) 
x_mean = (x_cv - mu)
x_cv = (x_cv - mu)/sigma
X_cv = poly(x_cv,5)


# In[26]:


# Model Evaluation - RMSE
def rmse(Y, Y_pred):
    rmse = np.sqrt(sum((Y - Y_pred) ** 2) / m)
    return rmse

# Model Evaluation - R2 Score
def r2_score(Y, Y_pred):
    mean_y = np.mean(Y)
    ss_tot = sum((Y - mean_y) ** 2)
    ss_res = sum((Y - Y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

Y_pred_cv = np.dot(X_cv, w_norm) + b_norm

print("RMSE")
print(rmse(y_cv, Y_pred_cv))
print("R2 Score")
print(r2_score(y_cv, Y_pred_cv))


# In[ ]:





# # Comparing data vs prediction

# In[27]:


plt.scatter(x_norm[:,0],y,marker='x',c='r')
plt.scatter(x_norm[:,0],Y_pred,marker='o',c='b')
plt.xlabel("A_norm")
plt.ylabel("label")
plt.title("A_norm vs label")
plt.legend(["data", "prediction"], loc ="lower right")
plt.show()


# In[28]:


plt.scatter(x_norm[:,1],y,marker='x',c='r')
plt.scatter(x_norm[:,1],Y_pred,marker='o',c='b')
plt.xlabel("B_norm")
plt.ylabel("label")
plt.title("B_norm vs label")
plt.legend(["data", "prediction"], loc ="lower right")
plt.show()


# In[29]:


plt.scatter(x_norm[:,2],y,marker='x',c='r')
plt.scatter(x_norm[:,2],Y_pred,marker='o',c='b')
plt.xlabel("C_norm")
plt.ylabel("label")
plt.title("C_norm vs label")
plt.legend(["data", "prediction"], loc ="lower right")
plt.show


# In[ ]:





# In[ ]:





# In[30]:


predict(i=111,w=w_norm,b= b_norm),y[111]


# In[ ]:





# # Prediction and saving test csv file

# In[31]:


#importing dataset
data_pred = pd.read_csv('polynomial_test_data.csv')
x_test= data_pred.iloc[:,1:4].values


# In[32]:


mu     = np.mean(x_test,axis=0)   
sigma  = np.std(x_test,axis=0) 
x_test_norm = (x_test - mu)/sigma
x_test_norm.shape


# In[33]:


X_test=poly(x_test_norm,5)


# In[34]:


y_test=np.dot(X_test, w_norm) + b_norm


# In[35]:


y_test.shape


# In[36]:


data_pred.insert(0,'label',y_test)


# In[37]:


#saving the dataframe as a csv file
data_pred.to_csv('polynomial_test_data_pred.csv',index=False)


# In[ ]:





# In[ ]:




