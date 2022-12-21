#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns


# In[3]:


data = pd.read_excel(r"E:\Math\TSF Internship\The Sparks Foundation Taks_Data\Task 1 data.xlsx")
data.head()


# In[6]:


data.plot()


# In[7]:


data.plot(kind='bar')


# In[15]:


sns.scatterplot(data['Hours'],data['Scores'] , marker='+', color = 'k')


# In[18]:


X= data['Hours']
y= data['Scores']


# In[21]:


sns.barplot(X,y)


# In[23]:


sns.boxenplot(X,y)


# In[39]:


sns.kdeplot(X,y)


# In[48]:


sns.lineplot(X,y)


# In[53]:


sns.boxenplot(X)


# In[56]:


sns.countplot(X)


# In[57]:


sns.countplot(y)


# In[58]:


sns.jointplot(X,y)


# In[69]:


sns.regplot(X,y , color = 'k', marker = '+')


# In[70]:


sns.relplot(X,y)


# In[77]:


sns.swarmplot(X,y)


# In[79]:


sns.violinplot(X,y)


# In[80]:


sns.barplot(X,y)


# In[82]:


sns.lineplot(X,y, color ='g')


# In[83]:


sns.scatterplot(X,y,marker = '+')


# In[89]:


data.info()


# In[97]:


data.plot(title='Hours/Scores graph', color = ['b','r'] )


# In[100]:


sns.distplot(y)


# In[101]:


sns.distplot(X)


# In[102]:


sns.displot(X)


# In[103]:


sns.displot(y)


# In[106]:


sns.ecdfplot(X)
sns.ecdfplot(y)


# In[124]:


import numpy as np
X=np.sort(X)
y=np.sort(y)


# In[125]:


sns.scatterplot(X,y)


# In[127]:


sns.regplot(X,y, color= 'k')


# In[ ]:




