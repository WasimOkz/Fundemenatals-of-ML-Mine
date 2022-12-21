#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv(r"E:\Math\2.Mine_Interest_\Data_Science_Course\Datasets\K_means clustering dataset unsupervised\Country-data.csv")
data.head()


# In[3]:


X = data.iloc[:,1:]
X


# In[6]:


# Dendromgram!

import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(X,method= 'ward'))
plt.title('Dendrogram')
plt.xlabel('Clusters')
plt.ylabel('Distances')


# In[11]:


# no of clusters = 4
from sklearn.cluster import AgglomerativeClustering

ag_model= AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'ward')


# In[12]:


ag_model.fit(X)


# In[14]:


y_pred = ag_model.fit_predict(X)
y_pred


# In[16]:


X['clusters'] = y_pred


# In[18]:


df1 = X[X.clusters == 0 ]
df2 = X[X.clusters == 1]
df3 = X[X.clusters == 2 ]
df4 = X[X.clusters == 3 ]


# In[19]:


X.columns


# In[35]:


# now visualize all the clusters

# plot these clusters on scatter plot! 
# between all
#markers=['^', 's', 'p', 'h', '8']

plt.scatter(df1[['inflation','gdpp']] ,df1[['health','income']] ,s = 150 , marker = 's', color ='red' , label = 'Clsuter_1')
plt.scatter(df2[['inflation','gdpp']] ,df2[['health','income']] ,s = 150 , marker = '*', color ='blue' , label = 'Clsuter_2')
plt.scatter(df3[['inflation','gdpp']] ,df3[['health','income']] ,s = 150 , marker = '.', color ='green' , label = 'Clsuter_3')
plt.scatter(df4[['inflation','gdpp']] ,df4[['health','income']] ,s = 150 , marker = '^', color ='yellow' , label = 'Clsuter_4')


plt.title('All Clusters!')
plt.xlabel('Inflation & Gdpp' )
plt.ylabel('Health & Income')
plt.legend()
plt.grid()
plt.show()


# In[38]:


# by k-means!

# Elbow method
from sklearn.cluster import KMeans
sse = []
for i in range(1,11):
    k_means = KMeans(n_clusters = i)
    k_means.fit(X)
    sse.append(k_means.inertia_)

plt.plot(range(1,11),sse)
plt.title('Title')
plt.xlabel('No. of clusters')
plt.ylabel('SSE')


# In[40]:


# no. of clusters = 4

k_means = KMeans(n_clusters =4 )
k_means.fit(X)


# In[43]:


y_pred = k_means.predict(X)

X['cluster'] = y_pred


# In[45]:


df1 = X[X.cluster == 0 ]
df2 = X[X.cluster == 1]
df3 = X[X.cluster == 2 ]
df4 = X[X.cluster == 3 ]


# In[46]:


# now visualize all the clusters

# plot these clusters on scatter plot! 
# between all
#markers=['^', 's', 'p', 'h', '8']

plt.scatter(df1[['inflation','gdpp']] ,df1[['health','income']] ,s = 150 , marker = 's', color ='red' , label = 'Clsuter_1')
plt.scatter(df2[['inflation','gdpp']] ,df2[['health','income']] ,s = 150 , marker = '*', color ='blue' , label = 'Clsuter_2')
plt.scatter(df3[['inflation','gdpp']] ,df3[['health','income']] ,s = 150 , marker = '.', color ='green' , label = 'Clsuter_3')
plt.scatter(df4[['inflation','gdpp']] ,df4[['health','income']] ,s = 150 , marker = '^', color ='yellow' , label = 'Clsuter_4')


plt.title('All Clusters!')
plt.xlabel('Inflation & Gdpp' )
plt.ylabel('Health & Income')
plt.legend()
plt.grid()
plt.show()


# In[52]:


from sklearn.preprocessing import *
get_ipython().run_line_magic('pinfo', 'StandardScaler')


# In[55]:


from sklearn.preprocessing import StandardScaler

st_sc = StandardScaler()

X.health = st_sc.fit_transform(X[['health']])
X.health


# In[ ]:


from sklearn.preprocessing import MinMaxScaler

mm_sc = MinMaxScaler()

X.income = st_sc.fit_transform(X[['income']])

