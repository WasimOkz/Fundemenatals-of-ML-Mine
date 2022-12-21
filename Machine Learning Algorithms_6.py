#!/usr/bin/env python
# coding: utf-8

# # Hierarchical Clustering!
Two methods of Hierarchical Clustering 
i)  Agglomerative Clustering
ii) Divisive Clustering
# In[3]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


# here in this method we have to find the best number of clusters in the dataset by using the method of-
# Dadogram. Take euclidian distance b/w to points and draw a bar plot for all the clusters.


# In[5]:


data = pd.read_csv(r"E:\Math\2.Mine_Interest_\Data_Science_Course\Datasets\for supervised and unsupervised\IRIS.csv")
data.head()


# In[7]:


from sklearn.preprocessing import LabelEncoder

lb_en = LabelEncoder()


# In[10]:


data.species = lb_en.fit_transform(data.species)
data.head()


# In[37]:


# Dandogram

import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(data, method='ward'))
plt.title('DendromGram')
plt.xlabel('No. of clusters', color ='black')
plt.ylabel('Distances', color = 'black')
plt.show()


# In[19]:


# here is the best number of cluster is '3'


# In[20]:


from sklearn.cluster import AgglomerativeClustering


# In[25]:


ag_cl = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
ag_cl.fit(data)


# In[28]:


y_pred = ag_cl.fit_predict(data)


# In[29]:


data['clusters'] = y_pred
data.head()


# In[31]:


df1 = data[data.clusters==0]
df2 = data[data.clusters==1]
df3 = data[data.clusters==2]


# In[45]:


sns.set(rc={'figure.figsize':(17,8)})


# In[46]:


# now visualize all the clusters

# plot these clusters on scatter plot! 
# between all

plt.scatter(df1[['species','petal_length']],df1[['species','petal_width']], s = 200, marker= '.' , c= 'g', label = 'Cluster_1')
plt.scatter(df2[['species','petal_length']],df2[['species','petal_width']], s = 200,  marker= '+' , c= 'red', label = 'Cluster_2')
plt.scatter(df3[['species','petal_length']],df3[['species','petal_width']], s = 200, marker= '*' , c= 'k', label = 'Cluster_3')


plt.title('All Clusters!')
plt.xlabel('species,  pt_len' , fontsize = 15 , color = 'blue')
plt.ylabel('species, pt_wd', fontsize = 15, color = 'blue')
plt.legend()
plt.show()


# In[ ]:


# now visualize all the clusters

# plot these clusters on scatter plot! 
# between all

plt.scatter(df1[['species','sepal_length','petal_length']],df1[['species','sepal_width','petal_width']],  marker= '.' , c= 'g', label = 'Cluster_1')
plt.scatter(df2[['species','sepal_length','petal_length']],df2[['species','sepal_width','petal_width']],  marker= '+' , c= 'b', label = 'Cluster_2')
plt.scatter(df3[['species','sepal_length','petal_length']],df3[['species','sepal_width','petal_width']],  marker= '*' , c= 'k', label = 'Cluster_3')

plt.scatter(cent[:,0], cent[:,1] , c = 'r', label = 'Centroid')

plt.title('All Clusters!')
plt.xlabel('species, sp_len, pt_len')
plt.ylabel('species, sp_wd , pt_wd')
plt.legend()
plt.show()


# In[47]:


sns.reset_orig()

