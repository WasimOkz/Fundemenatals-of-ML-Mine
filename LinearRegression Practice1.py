#!/usr/bin/env python
# coding: utf-8

# In[64]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[65]:


data = pd.read_csv("E:\Math\CESD_Data Analytics\covid_19_clean_complete(csv).csv")
data.head()


# In[66]:


X = np.array(data['Confirmed']).reshape(-1,1)
y =np.array(data['Deaths']).reshape(-1,1)


# In[67]:


plt.scatter(X,y, marker = '+')


# In[68]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y , test_size = 0.2)


# In[69]:


from sklearn.linear_model import LinearRegression
lnr = LinearRegression()


# In[70]:


lnr.fit(X,y)


# In[71]:


lnrp = lnr.predict(X)


# In[72]:


plt.plot(X, lnrp , 'b-')
plt.scatter(X, y , marker = '+')


# In[73]:


y_pred = lnr.predict(x_test)
for i in range(len(y_pred)):
    print(i)


# In[74]:


from sklearn.linear_model import LinearRegression
lng = LinearRegression()


# In[75]:


import numpy as np
import matplotlib.pyplot as plt
x = np.array([1,3,5,7,9,11,13]).reshape(-1,1)
y = np.array([22000,21000,20000,19000,18000,17000,16000]).reshape(-1,1)


# In[76]:


plt.scatter(x,y,c='r', marker = '+')


# In[77]:


lng.fit(x,y)


# In[78]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,ytest = train_test_split(x,y, test_size = 0.1)


# In[79]:


lngp = lng.predict(x_train)
lngpp= lng.predict(x)


# In[80]:


print(x_train)


# In[81]:


plt.plot(x_train, lngp , 'g-')
plt.scatter(x,y,c='r' ,marker = '+')


# In[82]:


lng.predict(np.array(15).reshape(-1,1))


# In[90]:


uni = data['WHO Region'].unique()
print(uni)


# In[101]:


plt.plot(uni,data['Date'].unique()[:6])


# In[128]:


date = data['Date'].unique(), data['Country/Region'].unique()
print(len(date[0]))
print(len(date[1]))


# In[ ]:





# In[ ]:




