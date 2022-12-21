#!/usr/bin/env python
# coding: utf-8

# # 1). Linear Regression!

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[65]:


data = pd.read_csv(r"E:\Math\2.Mine_Interest_\Data_Science_Course\Datasets\Linear Regression dataset\housing.csv")
data.head()


# In[66]:


data.info()


# In[3]:


data.isnull().sum()


# In[4]:


mis_val = data['total_bedrooms'].mean()
mis_val


# In[5]:


data['total_bedrooms'].fillna(mis_val, inplace = True)


# In[6]:


data.isnull().sum()


# In[7]:


data['ocean_proximity'].nunique()


# In[8]:


values, labels = pd.factorize(data['ocean_proximity'])
values


# In[9]:


data['ocean_proximity'] = values
data.head()


# In[10]:


X = data.iloc[:,:-1]
y = data.iloc[:,-1]

display(X)
display(y)


# In[11]:


# train_test_split/validation

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y, test_size = 0.2 , random_state= 0)


# In[12]:


# linear Regression

from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()


# In[13]:


# training model
lr_model.fit(x_train,y_train)


# In[14]:


# Train accuracy

lr_model.score(x_train,y_train)


# In[15]:


# Test accuracy

lr_model.score(x_test,y_test)


# In[16]:


y_pred = lr_model.predict(x_test)
from sklearn import metrics

print('Mean Squared Error:', metrics.mean_squared_error(y_test,y_pred))


# In[17]:


# Applying Decisin Tree Classifier

from sklearn.tree import DecisionTreeClassifier

dc_model = DecisionTreeClassifier()


# In[18]:


dc_model.fit(x_train,y_train)


# In[19]:


dc_model.score(x_train,y_train)


# In[20]:


dc_model.score(x_test,y_test)


# In[21]:


y_pred = dc_model.predict(x_test)

from sklearn import metrics

print('Mean Squared Error:', metrics.mean_squared_error(y_test,y_pred))


# In[22]:


x_test


# In[23]:


y_test


# In[24]:


# Prediction is Absolutely good data of row numb. 6655
name = [0]

dc_model.predict(pd.DataFrame({'longitude':-118.13, 'latitude':34.16, 'housing_median_age':33.0	, 'total_rooms':2682.0,
       'total_bedrooms':716.0, 'population':2050.0, 'households':692.0, 'median_income':2.4817,'median_house_value':169500.0}, index= name))


# In[25]:


sns.countplot(y_pred, color = 'r')
plt.title('y_pred count plot')


#  # 2). Logistic Regression!

# In[26]:


data = pd.read_csv(r"E:\Math\2.Mine_Interest_\Data_Science_Course\Datasets\Logistic Regression dataset\heart_2020_cleaned.csv")
data.head()


# In[27]:


# data null values checking()
data.isnull().sum()


# In[28]:


# data info()
data.info()


# In[29]:


# train_test_split/validation

values,labels = pd.factorize(data['SkinCancer'])
data['SkinCancer'] = values


# In[30]:


from sklearn.preprocessing import LabelEncoder

le_hd = LabelEncoder()
le_sk = LabelEncoder()
le_al = LabelEncoder()
le_st = LabelEncoder()


# In[31]:


data['HeartDisease']    = le_hd.fit_transform(data['HeartDisease'])
data['Smoking']         = le_sk.fit_transform(data['Smoking'])
data['AlcoholDrinking'] = le_al.fit_transform(data['AlcoholDrinking'])
data['Stroke']          = le_st.fit_transform(data['Stroke'])


# In[32]:


display(data['Stroke'] ,data['AlcoholDrinking'],data['Smoking'] , data['HeartDisease']    )


# In[33]:


X = data.iloc[:,:5]
y = data.iloc[:,-1]
display(X, 'if skincancer is  yes:0 and No:1', y)


# In[34]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,y, test_size = 0.25)


# In[35]:


from sklearn.linear_model import LogisticRegression

lgr_model  = LogisticRegression()


# In[36]:


lgr_model.fit(x_train,y_train)


# In[37]:


# prediction

y_pred = lgr_model.predict(x_test)

df = pd.DataFrame({'Actucal value':y_test, 'Predicted value':y_pred})
df


# In[38]:


# train and test accuracy

print('Training Score of the model:',lgr_model.score(x_train,y_train))
print('Testing Score of the model:', lgr_model.score(x_test,y_test))


# In[39]:


# checking the goodness of the model

from sklearn import metrics

print("R square score of the model:", metrics.r2_score(y_test,y_pred))
print("Accuracy score of the model:", metrics.accuracy_score(y_test,y_pred))


# # 3). Decision Tree Regression and Classification

# In[40]:


data  = pd.read_csv(r"E:\Math\2.Mine_Interest_\Data_Science_Course\Datasets\Decision Tree_Random Forest_SVM classification and clustering dataset\penguins_size.csv")
data.head()


# In[41]:


print(data['species'].unique())
print(data['island'].unique())


# In[42]:


# use labelencoder to change the strings into values

from sklearn.preprocessing import LabelEncoder

le_sp = LabelEncoder()
le_is = LabelEncoder()


# In[43]:


data['species'] = le_sp.fit_transform(data['species'])
data['island']  = le_is.fit_transform(data['island'])
display(data['species'], data['island'])


# In[44]:


X = data.iloc[:,:-1]
y = data.iloc[:,-1]
display(X,y)


# In[45]:


# train_test_split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,y, test_size = 0.25)


# In[46]:


print(data.isnull().sum())


# In[47]:


data.info()


# In[ ]:





# In[48]:


# data cleaning
data['culmen_length_mm'].fillna(data['culmen_length_mm'].mean(), inplace = True)
data['culmen_depth_mm'].fillna(data['culmen_depth_mm'].mean() , inplace = True)
data['flipper_length_mm'].fillna(data['flipper_length_mm'].mean(), inplace = True)
data['body_mass_g'].fillna(data['body_mass_g'].mean() , inplace = True)

data.isnull().sum()


# In[49]:


data['sex'].fillna(method = 'bfill' , inplace = True)
data['sex'].isnull().sum()


# In[50]:


display(data.head())
display(data.isnull().sum())


# In[51]:


values, labels = pd.factorize(data['sex'])
data['sex'] = values


# In[52]:


# train_test_split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,y, test_size = 0.25)


# In[53]:


# model implementation

from sklearn.tree import DecisionTreeClassifier

dc_model = DecisionTreeClassifier()


# In[57]:


data['culmen_length_mm'].convert_dtypes(float)


# In[58]:


data['culmen_depth_mm '].convert_dtypes(float)


# In[ ]:


data['flipper_length_mm '].convert_dtypes(float)


# In[ ]:


data['body_mass_g'].convert_dtypes(float)


# In[59]:


# dc_model.fit(x_train,y_train)

