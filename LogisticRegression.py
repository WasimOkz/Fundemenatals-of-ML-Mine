#!/usr/bin/env python
# coding: utf-8

# 
# # Logistic Regression 
# # for Classification type problems  , like {yes,no / True,False / 0,1 etc.}

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


x = [1,2,3,4,5,6,7,8,9, 50,60,80,100]
y = [False,True,True,False,True,False,False,False,True,False,True, True,False]


# In[3]:


plt.scatter(x,y , marker= '+', c = 'r')


# In[4]:


from sklearn.model_selection import train_test_split


# In[5]:


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.1)


# In[6]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[7]:


model.fit(np.array(x_train).reshape(-1,1),y_train)


# In[8]:


model.score(np.array(x_test).reshape(-1,1),np.array(y_test).reshape(-1,1))


# In[9]:


model.predict_proba(np.array(x_test).reshape(-1,1))


# In[10]:


modp = model.predict(np.array(x_train).reshape(-1,1))
modp


# In[11]:


model.predict(np.array(np.linspace(1,100)).reshape(-1,1))


# In[12]:


print(np.linspace(1,100))


# # checking whether the animal is dog or not ( one-vs-rest{ovr})

# In[13]:




cls = np.array([True,False,False,True,False,False,False,False,True,False]).reshape(-1,1)
res = ['dog','cat','fish','dog','cow','deer','fish','horse','dog','sheep']


# In[14]:


mod = LogisticRegression()


# In[15]:


cls_train,cls_test,res_train,res_test= train_test_split(cls,res,test_size= 0.2)


# In[16]:


mod.fit(cls_train,res_train)


# In[17]:


mod.predict(np.array(False).reshape(-1,1))


# In[18]:


print(mod.predict(cls_train),res_train)
print(mod.predict(cls_test),res_test)


# In[19]:


mod.predict_proba(cls)


# In[20]:


mod.score(cls_train,res_train)


# In[21]:


mod.score(cls_test,res_test)


# # #Age_bought insurance, yes=1,no=0

# In[22]:




age = [23,26,76,43,22,49,40,51,18,50,89,30]
b_in= [0,0,1,1,0,1,1,1,0,1,1,1]


# In[23]:


age_train,age_test,b_in_train,b_in_test = train_test_split(age,b_in,test_size=0.1)


# In[24]:


modl = LogisticRegression()


# In[25]:


modl.fit(np.array(age_train).reshape(-1,1),b_in_train)


# In[26]:


#Actual Data
data= pd.DataFrame({'Age':age,'Budget Insurance':b_in})
data.head()


# In[27]:


#Trained Data
df= pd.DataFrame({"Trained_age":age_train,'Trained_budg_insu.':b_in_train})
df.head()


# In[28]:


# Test Data
dff = pd.DataFrame({'Test_age':age_test,'test_budg_ins':b_in_test})
display(dff)


# In[29]:


# predictions through 'trained data'
print(modl.predict(np.array(age_train).reshape(-1,1)))
print(age_train)


# In[30]:


# predictions through test data;
print(modl.predict(np.array(age_test).reshape(-1,1)))
print(age_test)


# In[31]:


#predictions through actual data
print(modl.predict(np.array(age).reshape(-1,1)))
print(age)


# In[32]:


plt.scatter(age_train,modl.predict(np.array(age_train).reshape(-1,1)), c='r',marker='+')


# In[33]:


# general predictions
if ((modl.predict(np.array(56).reshape(-1,1)))==np.array(1)):
    print("Yes the person is eligable")

else:
    print('Nope this person is not eligable becaue his/her age is less than 30')


# In[34]:


# general predictions
if ((modl.predict(np.array(26).reshape(-1,1)))==np.array(1)):
    print("Yes the person is eligable")

else:
    print('Nope this person is not eligable becaue his/her age is less than 30')


# In[35]:


# score of the model through test
modl.score(np.array(age_test).reshape(-1,1),np.array(b_in_test).reshape(-1,1))


# In[36]:


# score of the model through training 
modl.score(np.array(age_train).reshape(-1,1),np.array(b_in_train).reshape(-1,1))


# In[37]:


modl.predict_proba(np.array(age).reshape(-1,1))


# # Graphs for practice

# In[40]:


plt.plot(modl.predict_proba(np.array(age).reshape(-1,1)))


# In[41]:


plt.scatter(np.array(age_train).reshape(-1,1),np.array(b_in_train).reshape(-1,1) , marker = 'd', c='g')


# In[42]:


plt.scatter(age_test,b_in_test, marker = "+", c='purple')


# In[43]:


from sklearn.metrics import confusion_matrix

cn_mt = confusion_matrix(y_test, y_pred)


# In[ ]:




