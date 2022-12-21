#!/usr/bin/env python
# coding: utf-8

# # data Prepocessing

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv(r"E:\Math\2.Mine_Interest_\Data_Science_Course\Datasets\Linear Regression dataset\housing.csv")
data.head()


# In[3]:


data.info()


# In[4]:


data.isnull().sum()


# In[5]:


from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values = "Nan", strategy = 'mean')


# In[36]:


data['ocean_proximity'].unique()


# In[17]:


# polynomial Regression

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree  =2)

"""
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(x_poly,y)
lr_model = LinearRegression()
lr_model.fit(x_poly,y)



"""
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()


# In[19]:


import numpy as np

X = np.array([1,2,3,4,8,9,10]).reshape(-1,1)
y = np.array([2,4,9,16,64,81,100]).reshape(-1,1)

X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly,y)


# In[20]:


lr_model.fit(X_poly,y)


# In[27]:


plt.scatter(X,y, marker = "+" , color = 'r')
plt.plot(X, lr_model.predict(X_poly), color = 'b')


# In[29]:


X_poly


# In[33]:


lr_model.predict(X_poly)


# In[ ]:





# In[ ]:




