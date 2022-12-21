#!/usr/bin/env python
# coding: utf-8

# # Muhammad Wasim
# # The Sparks Foundation Internship Batch July- Aug
# # Function/Domain :- Data Science and Bussiness Analytics

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # Extracting Data form Excel file, (downloaded data)

# In[2]:


data = pd.read_excel(r"E:\Math\TSF Internship\The Sparks Foundation Taks_Data\Task 1 data.xlsx")
data


# # locating data 

# In[3]:


X = data.iloc[:,:-1].values
y = data.iloc[:,1].values


# # Visualizing Given Data on a scatter graph

# In[4]:


plt.scatter(X,y , marker = '+', color = 'r')


# # Training Data

# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


x_train,x_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=0)


# # Applying Simple Linear Regression Model

# In[7]:


from sklearn.linear_model import LinearRegression

lnr= LinearRegression()


# # Fitting data

# In[8]:


lnr.fit(x_train,y_train)


# #  prediction Method

# In[9]:


lnrp = lnr.predict(x_train)


# # Visualization of predicted data vs Actual Data

# In[10]:


plt.plot(x_train, lnrp , 'g-')
plt.scatter(X, y, marker= '+' , color = 'r')
plt.title("Prediction of Scores on basis of Study Hours")
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Scores')
plt.show()


# # Predicting values when student studied for 9.2 Hours

# In[15]:


pr = np.array(9.25).reshape(-1,1)
print("If student studied for  9.25 hours then his/her socore will be :", lnr.predict(pr))


# # Making Predictions

# In[ ]:


print(x_test)


# In[ ]:


y_pred = lnr.predict(x_test)
y_pred


# In[ ]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# In[ ]:




