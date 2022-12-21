#!/usr/bin/env python
# coding: utf-8

# # Multiple Linear Regression!

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv(r"E:\Math\2.Mine_Interest_\Data_Science_Course\Datasets\Svm dataset\IRIS.csv")
data.head()


# In[3]:


data.species.nunique()


# In[4]:


dummies = pd.get_dummies(data.species)
dummies


# In[5]:


merged_df = pd.concat([data,dummies] , axis = 1)
merged_df


# In[6]:


data.species.unique()


# In[7]:


# we have to drop any two columns one the categorical_values column and the other one from dummies anyone from it!

final = merged_df.drop(['species', 'Iris-setosa'], axis = 1)
final


# In[8]:


X = final.iloc[:,1:]
y = final.sepal_length

display(X,y)


# In[9]:


from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()


# In[10]:


lr_model.fit(X,y)


# In[11]:


y_pred = lr_model.predict(X)
y_pred


# In[12]:


df = pd.DataFrame({'Actual Values of length':y, 'Predicted values of length':y_pred})
df


# In[13]:


lr_model.score(X,y)


# # dummies variable by LabelEncoder and !

# In[14]:


from sklearn.preprocessing import LabelEncoder

label_En = LabelEncoder()


# In[15]:


data.head()


# In[16]:


data.species.unique()


# In[17]:


data.species = label_En.fit_transform(data.species)
data


# In[18]:


X = data.iloc[:,:-1]
y = data.petal_width
display(X,y)


# In[19]:


lr_model = LinearRegression()
lr_model.fit(X,y)


# In[20]:


# 'Iris-setosa': 0 ,  'Iris-versicolor': 1 ,  'Iris-virginica': 2  !

y_pred = lr_model.predict(X)

df = pd.DataFrame({'Actual petal_width':y, 'Predicted petal_width':y_pred})
df


# In[43]:


from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(X)


# In[21]:


plt.scatter(X.species,y)
plt.xlabel('Counrtries specie code', color ='blue')
plt.ylabel('Petal_width', color ='blue')


# In[22]:


sns.pairplot(data)


# In[37]:


sns.heatmap(X)


# In[39]:


sns.heatmap(df)


# In[25]:


sns.swarmplot(x='sepal_length', y = 'sepal_width', data = data ,)


# In[26]:


sns.set(rc={'figure.figsize':(15,8)})
sns.swarmplot(data = data)


# In[27]:


sns.reset_orig()


# In[28]:


def graphs_fun(data):
    list = []
    for i in data.columns:
        list.append(i)
        
    for j in range(len(list)):
        plt.plot(data.columns[j+1])


# In[29]:


sns.barplot(data.species,data.sepal_length)


# In[30]:


sns.barplot(data.species,data.sepal_width)


# In[31]:


sns.barplot(data.species,data.petal_length)


# In[32]:


sns.barplot(data.species,data.petal_width)


# In[33]:


sns.distplot(data.petal_length)


# In[34]:


sns.barplot(data.sepal_length,data.sepal_length)


# In[35]:


sns.scatterplot(data.sepal_length,data.sepal_width)


# In[ ]:




