#!/usr/bin/env python
# coding: utf-8

# In[9]:


name = 'King'
print('This is the oldest version of {}'.format(name))


# In[15]:


print("My name is {} and I am {} years old".format("'Muhammad Wasim'","'19'"))


# In[16]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[27]:


data = pd.read_excel(r"E:\Math\TSF Internship\The Sparks Foundation Taks_Data\Task 1 data.xlsx")


# In[29]:


data.head()


# In[38]:


X = data.iloc[:,-1].values
y = data.iloc[:,1].values


# In[39]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# In[40]:


from sklearn.linear_model import LinearRegression

lnr= LinearRegression()


# In[42]:


lnr.fit(np.array(x_train).reshape(-1,1),y_train)


# In[44]:


lnrp = lnr.predict(np.array(x_train).reshape(-1,1))


# In[45]:


print(x_test)


# In[47]:


y_pred = lnr.predict(np.array(x_test).reshape(-1,1))
y_pred


# In[48]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# In[50]:


from sklearn.linear_model import metric


# In[ ]:




