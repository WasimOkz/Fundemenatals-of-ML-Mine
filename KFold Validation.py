#!/usr/bin/env python
# coding: utf-8

# In[1]:


# if we use the kfold method by this method we can find that which model is best if we have diff. models for our problem!


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


data = pd.read_csv(r"E:\Math\2.Mine_Interest_\Data_Science_Course\Datasets\Svm dataset\IRIS.csv")
data.head()


# In[4]:


from sklearn.preprocessing import LabelEncoder

lb_en = LabelEncoder()


# In[5]:


data.species = lb_en.fit_transform(data.species)
data.head()


# In[6]:


X = data.iloc[:,1:]
y = data.species


# In[7]:


# importing k-fold class


# In[8]:


# from sklearn.model_selection import KFold

# kf =KFold(n_splits = 10)


# In[9]:


# or to use this one. it is more effiecient than simple KFold 

# from sklearn.model_selection import StratifiedKFold

# st_kf = StratifiedKFold(n_splits = 10)


# In[10]:


# to check the score of the models that which one is best than use this method!


# In[11]:


from sklearn.model_selection import cross_val_score


# In[12]:


# importing different models

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# In[13]:


print(cross_val_score(SVC(), X, y))
print(cross_val_score(RandomForestClassifier(n_estimators = 40), X, y))
print(cross_val_score(DecisionTreeClassifier(), X, y))


# In[14]:


# find the avg of these values and then select the model with greater avg.


# In[15]:


print('SVM model avg score:',cross_val_score(SVC(), X, y).mean())
print('DecisionTreeClassifier model avg score:',cross_val_score(DecisionTreeClassifier(), X, y).mean())
print('RandomForestClassifier model avg score:',cross_val_score(RandomForestClassifier(n_estimators = 40), X, y).mean())


# In[ ]:




