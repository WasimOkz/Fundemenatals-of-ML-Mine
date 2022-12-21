#!/usr/bin/env python
# coding: utf-8

# # SVM Algorithm Practics

# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[2]:


data = pd.read_csv(r"E:\Math\2.Mine_Interest_\Data_Science_Course\Datasets\Svm dataset\IRIS.csv")
data.head()


# In[3]:


data.isnull().sum()


# In[4]:


from sklearn.preprocessing import LabelEncoder

lb_En  = LabelEncoder()


# In[5]:


data.species = lb_En.fit_transform(data.species)
data.head()


# In[10]:


df1 = data[data.species ==0]
df2 = data[data.species ==1]
df3 = data[data.species ==2]


# In[12]:


plt.scatter(df1.sepal_length,df1.sepal_width, color = 'red', marker = "+")
plt.scatter(df2.sepal_length,df2.sepal_width, color = 'green', marker = ".")
plt.xlabel('sepal length')
plt.ylabel('sepal width')


# In[13]:


plt.scatter(df1.petal_length,df1.petal_width, color = 'red', marker = "+")
plt.scatter(df2.petal_length,df2.petal_width, color = 'green', marker = ".")
plt.xlabel('petal length')
plt.ylabel('petal width')


# In[16]:


X = data.iloc[:,1:]
y = data.iloc[:,0:1]       # or data.species 
display(X,y)


# In[17]:


from sklearn.model_selection import train_test_split

x_train,x_test, y_train,y_test = train_test_split(X,y, test_size= 0.2)


# In[45]:


from sklearn.svm import SVC

sv_model = SVC(C=10)


# In[46]:


sv_model.fit(x_train,y_train)


# In[47]:


y_pred = sv_model.predict(x_test)


# In[48]:


# accuracy

sv_model.score(x_train,y_train)


# In[49]:


sv_model.score(x_test,y_test)


# In[50]:


sns.pairplot(data)


# In[ ]:




