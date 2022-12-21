#!/usr/bin/env python
# coding: utf-8

# In[1]:


# k_means clustering
'''


for this method first of all we need to find the number of clusters and to find this number we have to use elbow method.
what is elbow methos?
elbow method is to draw the distances from one data point to the all data points and then draw these distances on a graph 
in decending order(the most distance will visualize first then 2nd and so on.). 

'''


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


data = pd.read_csv(r'E:\Math\2.Mine_Interest_\Data_Science_Course\Datasets\K_means clustering dataset unsupervised\Country-data.csv')
data.head()


# In[4]:


X = data.iloc[:,5:7]
X


# In[5]:


# using the elbow method to find the number of optimal clusters

from sklearn.cluster import KMeans


# In[6]:


wcss = []
for i in range(1,11):
    k_means = KMeans(n_clusters = i , init = 'k-means++')
    k_means.fit(X)
    wcss.append(k_means.inertia_)

plt.plot(range(1,11),wcss)
plt.title("The Elbow Methos")
plt.xlabel("No. of Clusters")
plt.ylabel("WCSS")
plt.show()


# In[7]:


# the graph is behaviour is same after '4' so no of clusters == 4


# In[8]:


# fitting k_means to the data set

k_means = KMeans(n_clusters = 4, init = 'k-means++')
k_means.fit(X)


# In[9]:


X


# In[10]:


y_kmeans = k_means.fit_predict(X)


# In[11]:


y_kmeans 


# In[12]:


# visualizing the clusters 

plt.scatter([X[y_kmeans == 0,0] , X[y_kmeans == 0 ,1]],   s=100 , c = 'red', label = 'Cluster_1')
plt.scatter(X[y_kmeans == 1,0] , X[y_kmeans == 1 ,1],   s=100 , c = 'blue', label = 'Cluster_2')
plt.scatter(X[y_kmeans == 2,0] , X[y_kmeans == 2 ,1],   s=100 , c = 'cyan', label = 'Cluster_3')
plt.scatter(X[y_kmeans == 3,0] , X[y_kmeans == 3 ,1],   s=100 , c = 'magneta', label = 'Cluster_4')
plt.scatter(k_means.cluster_centers_[:, 0] , k_means.clusters_centers_[:,1] ,   s=200 , c= 'yellow', label = 'Centroids')
plt.title('Clusters of Country_data')
plt.xlabel('Income of the country')
plt.ylabel('Inflaion rate of the country')
plt.legend()
plt.show()


# In[13]:


plt.(X[y_kmeans == 1] , c = 'blue', label = 'Cluster_2')


# In[14]:


plt.scatter(k_means.cluster_centers_[:, 0] , k_means.cluster_centers_[:,1] , marker = '+' , s=200 , c= 'yellow', label = 'Centroids')
plt.title('Clusters of Country_data')
plt.xlabel('Income of the country')
plt.ylabel('Inflaion rate of the country')
plt.legend()
plt.show()


# In[15]:


X['cluster'] = y_kmeans


# In[16]:


X


# In[17]:


X1 = X[X.cluster==0] 
X2 = X[X.cluster==1] 
X3 = X[X.cluster==2] 
X4 = X[X.cluster==3]

plt.scatter(X1.income,X1.inflation, color ='green', s= 100 , label = 'Cluster 1')
plt.scatter(X2.income,X2.inflation, color ='blue' ,s= 100, label = 'Cluster 2')
plt.scatter(X3.income,X3.inflation, color ='cyan' ,s= 100, label = 'Cluster 3')
plt.scatter(X4.income,X4.inflation, color ='red' ,s= 100, label = 'Cluster 4')

plt.title('Clusters of Country_data')
plt.xlabel('Income of the country')
plt.ylabel('Inflaion rate of the country')
plt.legend()
plt.show()


# In[18]:


X1 = X[X.cluster==0] 
X2 = X[X.cluster==1] 
X3 = X[X.cluster==2] 
X4 = X[X.cluster==3]

plt.scatter(X1.income,X1.inflation, color ='green', s= 100 , label = 'Cluster 1')
plt.scatter(X2.income,X2.inflation, color ='blue' ,s= 100, label = 'Cluster 2')
plt.scatter(X3.income,X3.inflation, color ='cyan' ,s= 100, label = 'Cluster 3')
plt.scatter(X4.income,X4.inflation, color ='red' ,s= 100, label = 'Cluster 4')

plt.scatter(k_means.cluster_centers_[:, 0] , k_means.cluster_centers_[:,1] , marker = '+' , s=200 , c= 'yellow', label = 'Centroids')


plt.title('Clusters of Country_data')
plt.xlabel('Income of the country')
plt.ylabel('Inflaion rate of the country')
plt.legend()
plt.show()


# In[19]:


from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()


# In[20]:


sc.fit(np.array(X.inflation).reshape(-1,1))
X.inflation = sc.transform(np.array(X.inflation).reshape(-1,1))


# In[21]:


sc.fit(np.array(X.income).reshape(-1,1))
X.income = sc.transform(np.array(X.income).reshape(-1,1))


# In[22]:


X


# In[23]:


X.drop('cluster', axis = 'columns')


# In[ ]:





# In[24]:


from sklearn.cluster import KMeans

k_means = KMeans(n_clusters = 4, init = 'k-means++')
k_means.fit(X)


# In[25]:


y_pred = k_means.fit_predict(X)
y_pred


# In[ ]:





# In[26]:


X['cluster'] = y_pred
X


# In[27]:


X1 = X[X.cluster==0] 
X2 = X[X.cluster==1] 
X3 = X[X.cluster==2] 
X4 = X[X.cluster==3]

plt.scatter(X1.income,X1.inflation, color ='green', s= 150 , label = 'Cluster 1', marker = '.')
plt.scatter(X2.income,X2.inflation, color ='blue' ,s= 150, label = 'Cluster 2', marker = '.')
plt.scatter(X3.income,X3.inflation, color ='cyan' ,s= 150, label = 'Cluster 3', marker = '.')
plt.scatter(X4.income,X4.inflation, color ='red' ,s= 150, label = 'Cluster 4' , marker = '.')

plt.scatter(k_means.cluster_centers_[:, 0] , k_means.cluster_centers_[:,1] , marker = '+' , s=150 , c= 'black', label = 'Centroids')


plt.title('Clusters of Country_data')
plt.xlabel('Income of the country')
plt.ylabel('Inflaion rate of the country')
plt.legend()
plt.show()


# # some more practice with new columns (child_mort , exports) 

# In[28]:


data.head()


# In[29]:


X = data.iloc[:,1:3]
X


# In[30]:


# first of all we have to apply the elbow method to find the number of clusters

X.isnull().sum()


# In[31]:


from sklearn.cluster  import KMeans

k_means = KMeans()


# In[32]:


wcss =[]
for i in range(1,11):
    k_means = KMeans(n_clusters = i)
    k_means.fit(X)
    wcss.append(k_means.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title("Elbow Method")
plt.xlabel('No. of Clusters')
plt.ylabel('WCSS')
plt.legend()
plt.show()


# In[33]:


# The curve is changing upto 5 so it has '5' clusters.


# In[34]:


k_means = KMeans(n_clusters = 5)
k_means.fit(X)


# In[35]:


y_pred = k_means.predict(X)
y_pred


# In[36]:


X['clusters'] = y_pred
X


# In[37]:


X1 = X[X.clusters == 0]
X2 = X[X.clusters == 1]
X3 = X[X.clusters == 2]
X4 = X[X.clusters == 3]
X5 = X[X.clusters == 4]


# In[38]:


# centriods

centroids = k_means.cluster_centers_
centroids


# In[39]:


# visualizing clusters and centroids

plt.scatter(X1.child_mort , X1.exports, c = 'red', label = 'Cluster 1', s = 150 , marker = '.')
plt.scatter(X2.child_mort , X2.exports, c = 'blue', label = 'Cluster 2' , s = 150 , marker= '.')
plt.scatter(X3.child_mort , X3.exports, c = 'orange', label = 'Cluster 3' , s = 150, marker= '.')
plt.scatter(X4.child_mort , X4.exports, c = 'black', label = 'Cluster 4', s = 150 , marker = '.')
plt.scatter(X5.child_mort , X5.exports, c = 'green', label = 'Cluster 5' , s = 150 , marker = '.')



plt.scatter(centroids[:,0], centroids[:,1], c = 'purple', label = 'Centroids', s = 300, marker = '+')


plt.title("Clusters and Centroids")
plt.xlabel('child_mort')
plt.ylabel('export')
plt.legend()
plt.show()


# In[40]:


print('Centroids of child_mort column:',centroids[:,0])


# In[41]:


print('Centroids of export  columns:',centroids[:,1])


# # new  ( health, imports)

# In[42]:


data.head()


# In[43]:


X = data.iloc[:,3:5]
X


# In[44]:


# Elbow method to find the no_of clusters


# In[45]:



from sklearn.cluster import KMeans

k_means = KMeans()


# In[46]:


wcss = []

for i in range(1,11):
    k_means = KMeans(n_clusters = i)
    k_means.fit(X)
    wcss.append(k_means.inertia_)
    
# ploting elbow graph!

plt.plot(range(1,11), wcss)
plt.title('Elbow Method')
plt.xlabel('No of clusters')
plt.ylabel('WCSS')
sns.set(rc= {'figure.figsize':(10,15)})


# In[47]:


# number of clusters = 4

k_means  = KMeans(n_clusters = 4)
k_means.fit(X)


# In[48]:


y_pred = k_means.predict(X)
y_pred


# In[49]:


X['Clusters'] = y_pred
X


# In[50]:


X1 = X[X.Clusters == 0]
X2 = X[X.Clusters == 1]
X3 = X[X.Clusters == 2]
X4 = X[X.Clusters == 3]


# In[51]:


# centroids_

centroids  = k_means.cluster_centers_
centroids


# In[52]:


sns.reset_orig()


# In[53]:


# visualizing clusters and centroids

plt.scatter(X1.health, X1.imports , c ='red', marker = '.', s = 200, label = 'Cluster 1')
plt.scatter(X1.health, X1.imports , c ='black', marker = '.', s = 200, label = 'Cluster 2')
plt.scatter(X1.health, X1.imports , c ='yellow', marker = '.', s = 200, label = 'Cluster 3')
plt.scatter(X1.health, X1.imports , c ='green', marker = '.', s = 200, label = 'Cluster 4')
# plt.scatter(X1.health, X1.imports , c ='blue', marker = '.', s = 200, label = 'Cluster 5')
# plt.scatter(X1.health, X1.imports , c ='cyan', marker = '.', s = 200, label = 'Cluster 6')

plt.scatter(centroids[:,0], centroids[:,1], c = 'purple', marker = '.', s = 250 , label = 'Centroids')

plt.title("Clusters of Health and Imports")
plt.xlabel('Health data')
plt.ylabel('Imports data')
plt.legend()
plt.show()


# In[54]:


print(len(X.Clusters==0))
print(len(X.Clusters==1))
print(len(X.Clusters==2))
print(len(X.Clusters==3))
print(len(X.Clusters==4))
print(len(X.Clusters==5))


# # code basics video data :)

# In[55]:


data = pd.read_excel(r"C:\Users\Hamzapc\Desktop\age_income_data.xlsx")
data.head()


# In[56]:


plt.scatter(data['Age'], data['income'])


# In[57]:


# apply elbow method to find the number of clusters!

from sklearn.cluster import KMeans


# In[58]:


wcss = []

for i in range(1,11):
    k_means  = KMeans(n_clusters = i)
    k_means.fit(data)
    wcss.append(k_means.inertia_)


# In[59]:


# visualizing elbow graph!

plt.plot(range(1,11), wcss ,  'r')
plt.title('Elbow Method')
plt.xlabel('No. of clusters')
plt.ylabel('WCSS')


# In[60]:


# number of clusters = 3

k_means = KMeans(n_clusters = 3)
k_means.fit(data)


# In[61]:


y_pred = k_means.predict(data)
y_pred


# In[62]:


data['Clusters_'] = y_pred
data


# In[63]:


X1 = data[data.Clusters_ == 0]
X2 = data[data.Clusters_ == 1]
X3 = data[data.Clusters_ == 2]


# In[64]:


# centroids!
centroids = k_means.cluster_centers_
centroids


# In[65]:


# visualization

plt.scatter(X1.Age , X1.income , color = 'blue', s = 200, marker = '.', label = 'Cluster 1')
plt.scatter(X1.Age, X1.income , color = 'red', s = 200, marker = '.', label = 'Cluster 2')

plt.scatter(centroids[:,0], centroids[:,1], color = 'green', s = 240, marker = '+', label = 'Centroids')

plt.title("Clusters visualizations")
plt.xlabel('Age')
plt.ylabel('Income')
plt.legend()
plt.show()


# In[66]:


wcss


# # k-means clustering using IRIS data_set

# In[67]:


data = pd.read_csv(r"E:\Math\2.Mine_Interest_\Data_Science_Course\Datasets\for supervised and unsupervised\IRIS.csv")
data.head()


# In[69]:


# convert string data into values!

from sklearn.preprocessing import LabelEncoder

lb_en = LabelEncoder()


# In[71]:


data.species = lb_en.fit_transform(data.species)
data.species


# In[72]:


from sklearn.cluster import KMeans


# In[75]:


# we have to find the number of clusters by elbow method! sse: sum of squared errors

sse = []
for i in range(1,11):
    
    k_means = KMeans(n_clusters = i)
    k_means.fit(data)
    sse.append(k_means.inertia_)

# visualize elbow method!

plt.plot(range(1,11),sse, c='b')
plt.title('Elbow method!')
plt.xlabel('No. of clusters')
plt.ylabel('SSE:sum of squared Errors!')
plt.show()


# In[76]:


# the best number is 3

k_means = KMeans(n_clusters = 3)
k_means.fit(data)


# In[77]:


y_pred = k_means.predict(data)
y_pred


# In[78]:


data['cluster'] = y_pred
data


# In[81]:


df1 = data[data.cluster==0]
df2 = data[data.cluster==1]
df3 = data[data.cluster==2]


# In[84]:


cent = k_means.cluster_centers_
cent


# In[85]:


# plot these clusters on scatter plot! 
# between sepal_length and width

plt.scatter(df1.sepal_length,df1.sepal_width,  marker= '.' , c= 'g', label = 'Cluster_1')
plt.scatter(df2.sepal_length,df2.sepal_width,  marker= '+' , c= 'b', label = 'Cluster_2')
plt.scatter(df3.sepal_length,df3.sepal_width,  marker= '*' , c= 'k', label = 'Cluster_3')


plt.scatter(cent[:,0], cent[:,1] , c = 'r', label = 'Centroid')

plt.legend()
plt.show()


# In[87]:


plt.scatter(df1[['species','sepal_length','petal_length']],df1[['species','sepal_width','petal_width']],  marker= '.' , c= 'g', label = 'Cluster_1')


# In[89]:


# plot these clusters on scatter plot! 
# between all

plt.scatter(df1[['species','sepal_length','petal_length']],df1[['species','sepal_width','petal_width']],  marker= '.' , c= 'g', label = 'Cluster_1')
plt.scatter(df2[['species','sepal_length','petal_length']],df2[['species','sepal_width','petal_width']],  marker= '+' , c= 'b', label = 'Cluster_2')
plt.scatter(df3[['species','sepal_length','petal_length']],df3[['species','sepal_width','petal_width']],  marker= '*' , c= 'k', label = 'Cluster_3')

plt.scatter(cent[:,0], cent[:,1] , c = 'r', label = 'Centroid')

plt.legend()
plt.show()


# In[91]:


from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()


# In[93]:


data.species = sc.fit_transform(data[['species']])
data.sepal_length = sc.fit_transform(data[['sepal_length']])
data.sepal_width = sc.fit_transform(data[['sepal_width']])
data.petal_length = sc.fit_transform(data[['petal_length']])
data.petal_width = sc.fit_transform(data[['petal_width']])
data


# In[101]:


data.drop('cluster', axis = 'columns', inplace = True)


# In[102]:


data


# In[103]:


k_means = KMeans(n_clusters = 3)
k_means.fit(data)


# In[104]:


y_pred = k_means.predict(data)
y_pred


# In[105]:


data['cluster'] = y_pred
data


# In[107]:


cent = k_means.cluster_centers_
cent


# In[108]:


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


# In[109]:


# drop the sepal length and width features and then draw


# In[111]:


X = data.drop(['sepal_length', 'sepal_width','cluster'], axis=1 )
X


# In[113]:


def elbow_method(data):
    sse = []
    for i in range(1,11):
    
        k_means = KMeans(n_clusters = i)
        k_means.fit(data)
        sse.append(k_means.inertia_)

    # visualize elbow method!

    plt.plot(range(1,11),sse, c='b')
    plt.title('Elbow method!')
    plt.xlabel('No. of clusters')
    plt.ylabel('SSE:sum of squared Errors!')
    plt.show()


# In[114]:


elbow_method(X)


# In[115]:


# best number is '3'. so use the old one

k_means.fit(X)


# In[116]:


y_pred = k_means.predict(X)

X['Cluster'] = y_pred
X


# In[119]:


df1 = X[X.Cluster ==0]
df2 = X[X.Cluster ==1]
df3 = X[X.Cluster ==2]


# In[120]:


cent = k_means.cluster_centers_
cent


# In[131]:


sns.set(rc={'figure.figsize':(15,10)})


# In[137]:


# now visualize all the clusters

# plot these clusters on scatter plot! 
# between all

plt.scatter(df1[['species','petal_length']],df1[['species','petal_width']], s = 200, marker= '.' , c= 'g', label = 'Cluster_1')
plt.scatter(df2[['species','petal_length']],df2[['species','petal_width']], s = 200,  marker= '+' , c= 'b', label = 'Cluster_2')
plt.scatter(df3[['species','petal_length']],df3[['species','petal_width']], s = 200, marker= '*' , c= 'k', label = 'Cluster_3')

plt.scatter(cent[:,0], cent[:,1] , c = 'red', s = 300, marker ='d', label = 'Centroid')

plt.title('All Clusters!')
plt.xlabel('species,  pt_len' , fontsize = 15 , color = 'blue')
plt.ylabel('species, pt_wd', fontsize = 15, color = 'blue')
plt.legend()
plt.show()


# In[138]:


sns.reset_orig()

