#!/usr/bin/env python
# coding: utf-8

# In[1]:


# gradient decent is an algorithm use to find the best fit line 


# In[2]:


# funtion of gradient decent
# m: slope/coefficient , b: intercept, cost_function is MSE: (1/n)*summision (y-y_pred)**2

def Gradient_Decent(x,y):
    m_curr = b_curr = 0
    iterations = 25000
    n = len(x)
    learning_rate = 0.001
    
    for i in range(iterations):
        y_pred = m_curr * x + b_curr
        cost = (1/n)*sum([val**2 for val in (y-y_pred)])
        md= -(2/n)*sum(x*(y-y_pred))
        bd= -(2/n)*sum(y-y_pred)
        
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        
        print('m:{} , b:{} , cost:{} , iterations:{} , '.format(m_curr,b_curr, cost, i))


# In[3]:


import numpy as  np


# In[4]:


x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])


# In[5]:


Gradient_Decent(x,y)


# In[6]:


# visualization of this method that how to fit the best line
import matplotlib.pyplot as plt


plt.scatter(x,y, c='red', marker = "+" )


# In[7]:


from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()


# In[8]:


lr_model.fit(np.array(x).reshape(-1,1),np.array(y).reshape(-1,1))


# In[9]:


lr_model.coef_ , lr_model.intercept_


# In[10]:


# wow it is the very best what I found these coeficient and intercept by Gradient_Decent method


# In[11]:


y_pred =lr_model.predict(np.array(x).reshape(-1,1))


# In[12]:


plt.scatter(x,y, c= 'r', marker = '+')
plt.plot(x,y_pred , c= 'black')


# # Pickle Method! or joblib has the same functionality as Pickle

# In[14]:


# This method is using for the purpose that if you once create a model and later you want to use the same method so you
# can just call this picle method and run it


# In[16]:


# pickle Method
import pickle

# now we are saving this model in a directory that's why wb :writting.
with open('model_pickle', 'wb') as f:
    pickle.dump(lr_model,f)


# In[18]:


with open('model_pickle', 'rb') as f:
    model_pickle  = pickle.load(f)


# In[20]:


model_pickle.predict(np.array(x).reshape(-1,1))


# In[24]:


# same for joblib!
# here it is not working try later


# In[ ]:




