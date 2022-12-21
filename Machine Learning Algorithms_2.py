#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv(r"E:\Math\2.Mine_Interest_\Data_Science_Course\Datasets\Logistic Regression dataset\heart_2020_cleaned.csv")
data.head()


# In[3]:


data.isnull().sum()


# In[4]:


data.info()


# In[5]:


# import LabelEncoder  changing object type to values[0,1,2..]
from sklearn.preprocessing import LabelEncoder
le_sm = LabelEncoder()
le_al = LabelEncoder()
le_st = LabelEncoder()
le_dw = LabelEncoder()
le_sx = LabelEncoder()
le_ac = LabelEncoder()
le_rc = LabelEncoder()
le_db = LabelEncoder()
le_pa = LabelEncoder()
le_gh = LabelEncoder()
le_as = LabelEncoder()
le_kd = LabelEncoder()
le_sc = LabelEncoder()


# In[6]:


data['Smoking'] = le_sm.fit_transform(data['Smoking'])
data['AlcoholDrinking'] = le_al.fit_transform(data['AlcoholDrinking'])
data['Stroke'] = le_st.fit_transform(data['Stroke'])
data['DiffWalking'] = le_dw.fit_transform(data['DiffWalking'])
data['Sex'] = le_sx.fit_transform(data['Sex'])
data['AgeCategory'] = le_ac.fit_transform(data['AgeCategory'])
data['Race'] = le_rc.fit_transform(data['Race'])
data['Diabetic'] = le_db.fit_transform(data['Diabetic'])
data['PhysicalActivity'] = le_pa.fit_transform(data['PhysicalActivity'])
data['GenHealth'] = le_gh.fit_transform(data['GenHealth'])
data['Asthma'] = le_as.fit_transform(data['Asthma'])
data['KidneyDisease'] = le_as.fit_transform(data['KidneyDisease'])
data['SkinCancer'] = le_sc.fit_transform(data['SkinCancer'])


# In[7]:


data.head()


# In[8]:


X  = data.iloc[:,1:]
y  = data.iloc[:,0]


# In[9]:


display(X,y)


# In[10]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,y, test_size = 0.25)


# In[11]:


from sklearn.ensemble import RandomForestClassifier

rc_model = RandomForestClassifier()


# In[12]:


rc_model.fit(x_train,y_train)


# In[13]:


# checking the train and test accuracy of the model

rc_model.score(x_train,y_train)


# In[14]:


rc_model.score(x_test,y_test)


# In[15]:


# predictions

y_pred = rc_model.predict(x_test)
df = pd.DataFrame({'Predicted Data':y_pred , 'Actual Data':y_test})
df


# In[16]:


# r square score
from sklearn import metrics

print('The R square score of the model is :', metrics.r2_score(y_test,y_pred))


# In[17]:


# for this we have to convert the string to values

le_yp = LabelEncoder()
le_yt = LabelEncoder()


df['Predicted Data'] = le_yp.fit_transform(df['Predicted Data'])
df['Actual Data'] = le_yt.fit_transform(df['Actual Data'])
df


# In[18]:


print('The R square score of the model is :', metrics.r2_score(df['Actual Data'], df['Predicted Data']))


# In[19]:


# mean square error of the model
print('The mean square error of the model is:', metrics.mean_squared_error(df['Actual Data'], df['Predicted Data']))


# # Now apply DecisionTreeClassifier

# In[20]:


from sklearn.tree import DecisionTreeClassifier

# dc_model = DecisionTreeClassifier(max_leaf_node = 5,random_state=0)
dc_model = DecisionTreeClassifier()


# In[21]:


print(x_train,'\n\n',y_train,'\n\n',x_test,'\n\n',y_test)


# In[22]:


dc_model.fit(x_train,y_train)


# In[23]:


print('Predicted Data by Decision Tree Classifier')
y_pred = dc_model.predict(x_test)
y_pred


# In[24]:


df = pd.DataFrame({'Predicted Data':y_pred, 'Actual Data':y_test})
df


# In[25]:


# score Accuracy

print('Training Score:',dc_model.score(x_train,y_train), '\nTesting Score:', dc_model.score(x_test,y_test))


# In[26]:


# for this we have to convert the string to values

le_yp = LabelEncoder()
le_yt = LabelEncoder()


df['Predicted Data'] = le_yp.fit_transform(df['Predicted Data'])
df['Actual Data'] = le_yt.fit_transform(df['Actual Data'])
df


# In[27]:


# r square and mean square error
print('R square score:',metrics.r2_score(df['Actual Data'], df['Predicted Data']))
print('Mean square error', metrics.mean_squared_error(df['Actual Data'], df['Predicted Data']))


# In[28]:


# visualize tree
from sklearn import tree

# tree.plot_tree(dc_model)   when leaf_nodes = 5  
# plt.show()


# In[ ]:





# # set two columns in the output variable and then try!

# In[29]:


data.head()


# In[30]:


X = data.iloc[:,1:-1]
y = data.iloc[:,-2::]


# In[31]:


print(X,y)


# In[32]:


# implementing the RandomForestClassifier Model

rc_model = RandomForestClassifier()
x_train,x_test,y_train,y_test = train_test_split(X,y, test_size = 0.2)


# In[33]:


rc_model.fit(x_train,y_train)


# In[34]:


# predictions
y_pred = rc_model.predict(x_test)
print(y_pred)


# In[35]:


# training and testing accuracy

print("Training Accuracy:",rc_model.score(x_train,y_train))
print('Testing Accuracy:',rc_model.score(x_test,y_test))


# In[36]:


# R square score and mean square error and mean absolute error
print('R square Score is:',metrics.r2_score(y_test,y_pred))
print('mean square error is:',metrics.mean_squared_error(y_test,y_pred))
print('mean absolute error is:',metrics.mean_absolute_error(y_test,y_pred))


# In[85]:


df = pd.DataFrame(y_pred, columns = ['Kidney_Diseas_Predicted', 'Skin_Cancer_Predicted'])

dff = pd.DataFrame(y_test)
display(dff,df )


# In[ ]:





# In[ ]:




