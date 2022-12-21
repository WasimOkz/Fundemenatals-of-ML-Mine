#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'Price']
df=pd.read_csv(r"C:\Users\Hamzapc\Downloads\archive\housing.csv",names=column_names,delimiter=r"\s+")
df.head() 


# In[3]:


X = df.iloc[:,:-1]
y = df.iloc[:,-1]
display(X.head())
display(y.head())


# In[4]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test  = train_test_split(X,y , test_size = 0.3)


# # Linear Regression

# In[5]:


from sklearn.linear_model import LinearRegression

ln = LinearRegression()


# In[6]:


ln.fit(X,y)


# In[7]:


# Training Accuracy on LR
ln.score(x_train,y_train)


# In[8]:


# Testing Accuracy

ln.score(x_test,y_test)


# In[9]:


# R-squared Error
y_pred = ln.predict(x_test)

from sklearn import metrics
print('R Squared Error:',
         metrics.r2_score(y_test,y_pred))


# In[10]:


# Mean-squared Error

y_pred = ln.predict(x_test)

from sklearn import metrics
print('Mean squared Error:', 
      metrics.mean_squared_error(y_test, y_pred))


# # Random Forest Regressor

# In[11]:


from sklearn.ensemble import RandomForestRegressor

rr = RandomForestRegressor()


# In[12]:


rr.fit(x_train,y_train)


# In[13]:


# Training Accuracy

rr.score(x_train,y_train)


# In[14]:


# Testing Accuracy

rr.score(x_test,y_test)


# In[15]:


# R-squared Error
y_pred = rr.predict(x_test)

from sklearn import metrics
print('R Squared Error:',
         metrics.r2_score(y_test,y_pred))


# In[16]:


# Mean-squared Error

# Mean-squared Error

y_pred = rr.predict(x_test)

from sklearn import metrics
print('Mean squared Error:', 
      metrics.mean_squared_error(y_test, y_pred))


# # SVM Regressor

# In[17]:


from sklearn.svm import SVR

sv = SVR()


# In[18]:


sv.fit(x_train,y_train)


# In[19]:


# Trainging Accuracy

sv.score(x_train,y_train)


# In[20]:


# Testing Accuracy
sv.score(x_test,y_test)


# In[21]:


# R-squared Error
y_pred = sv.predict(x_test)

from sklearn import metrics
print('R Squared Error:',
         metrics.r2_score(y_test,y_pred))


# In[22]:


# Mean-squared Error

# Mean-squared Error

y_pred = sv.predict(x_test)

from sklearn import metrics
print('Mean squared Error:', 
      metrics.mean_squared_error(y_test, y_pred))


# # Gradient Boost Regressor

# In[23]:


from sklearn.ensemble import GradientBoostingRegressor

gr = GradientBoostingRegressor()


# In[24]:


gr.fit(x_train,y_train)


# In[25]:


# Training Accuracy

gr.score(x_train, y_train)


# In[26]:


# Testing Accuracy

gr.score(x_test,y_test)


# In[27]:


# R-squared Error
y_pred = gr.predict(x_test)

from sklearn import metrics
print('R Squared Error:',
         metrics.r2_score(y_test,y_pred))


# In[28]:


# Mean-squared Error

# Mean-squared Error

y_pred = gr.predict(x_test)

from sklearn import metrics
print('Mean squared Error:', 
      metrics.mean_squared_error(y_test, y_pred))

