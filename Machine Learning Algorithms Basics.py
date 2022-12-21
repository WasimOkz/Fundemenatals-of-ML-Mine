#!/usr/bin/env python
# coding: utf-8

# # Regression!  cont.data
# Linear Regression

from sklearn.linear_model import LinearRegression

ln_model = LinearRegression()

"""
from sklear.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size= 0.5)
ln_model.fit(x_train,y_train)
ln_model.score(x_train,y_train)
y_pred = ln_model.predict(x_test)


from sklear import metrics
print("Mean square/absolute error:", metrics.mean_square/absolute_error(y_test,y_pred))
print("R Squared Erro:",metrics.r2_score(y_test,y_pred)) # greater the value of R the most the model best fits.


"""# polynomial Regression

"""



from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree  =2)


X_poly = poly_reg.fit_transform(X)
poly_reg.fit(x_poly,y)
lr_model = LinearRegression()
lr_model.fit(x_poly,y)


from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()



import numpy as np

X = np.array([1,2,3,4,8,9,10]).reshape(-1,1)
y = np.array([2,4,9,16,64,81,100]).reshape(-1,1)

X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly,y)


lr_model.fit(X_poly,y)

plt.scatter(X,y, marker = "+" , color = 'r')
plt.plot(X, lr_model.predict(X_poly), color = 'b')
"""# Decision Tree Regressor

from sklearn.tree import DecisionTreeRegressor

dr_model = DecisionTreeRegressor(max_leaf_nodes = 5, random_state=0)

"""
from sklear.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size= 0.5)
dr_model.fit(x_train,y_train)
dr_model.score(x_train,y_train)
y_pred = dr_model.predict(x_test)


from sklear import metrics
print("Mean square/absolute error:", metrics.mean_square/absolute_error(y_test,y_pred))
print("R Squared Erro:",metrics.r2_score(y_test,y_pred)) # greater the value of R the most the model best fits.

# to visualize the tree

from sklearn import tree
tree.plot_tree(dr_model)
plt.show()
"""# lasso : L1 Regression
# Ridge : L2 Regression 

from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

ls_model = Lasso(alpha = 50)
rd_model = Ridge(alpha = 50)
"""
from sklear.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size= 0.5)

ls_model.fit(x_train,y_train)
rd_model.fit(x_train,y_train)

ls_model.score(x_train,y_train)
rd_model.score(x_train,y_train)

y_pred = ls_model.predict(x_test)
y_pred = rd_model.predict(x_test)

from sklear import metrics
print("Mean square/absolute error:", metrics.mean_square/absolute_error(y_test,y_pred))
print("R Squared Erro:",metrics.r2_score(y_test,y_pred)) # greater the value of R the most the model best fits.


"""# Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor

rr_modle = RandomForestRegressor(n_estimators=10) # n_estimators = 10 means 10 trees :)

"""
from sklear.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size= 0.5)
rr_model.fit(x_train,y_train)
rr_model.score(x_train,y_train)
y_pred = rr_model.predict(x_test)


from sklear import metrics
print("Mean square/absolute error:", metrics.mean_square/absolute_error(y_test,y_pred))
print("R Squared Erro:",metrics.r2_score(y_test,y_pred)) # greater the value of R the most the model best fits.


"""# Support Vector (SV) Regressor

from sklearn.svm import SVR

sr_model = SVR()

"""
from sklear.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size= 0.5)
sr_model.fit(x_train,y_train)
sr_model.score(x_train,y_train)
y_pred = sr_model.predict(x_test)


from sklear import metrics
print("Mean square/absolute error:", metrics.mean_square/absolute_error(y_test,y_pred))
print("R Squared Erro:",metrics.r2_score(y_test,y_pred)) # greater the value of R the most the model best fits.


"""# Gradient Boosting Regressor

from sklearn.ensemble import GradientBoostingRegressor

gr_model = GradientBoostingRegressor()

"""
from sklear.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size= 0.5)
gr_model.fit(x_train,y_train)
gr_model.score(x_train,y_train)
y_pred = gr_model.predict(x_test)


from sklear import metrics
print("Mean square/absolute error:", metrics.mean_square/absolute_error(y_test,y_pred))
print("R Squared Erro:",metrics.r2_score(y_test,y_pred)) # greater the value of R the most the model best fits.


"""# ANN Regressor
# # Classification! catogorial/discrete(yes/no) type data
# Logistic Regression

from sklearn.linear_model import LogisticRegression

lgr_model= LogisticRegression()

"""
from sklear.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size= 0.5)
lgr_model.fit(x_train,y_train)
lgr_model.score(x_train,y_train)
y_pred = lgr_model.predict(x_test)


from sklear import metrics
print("Mean square/absolute error:", metrics.mean_square/absolute_error(y_test,y_pred))
print("R Squared Erro:",metrics.r2_score(y_test,y_pred)) # greater the value of R the most the model best fits.


"""# Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

dc_model= DecisionTreeClassifier()

"""
from sklear.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size= 0.5)
dc_model.fit(x_train,y_train)
dc_model.score(x_train,y_train)
y_pred = dc_model.predict(x_test)


from sklear import metrics
print("Mean square/absolute error:", metrics.mean_square/absolute_error(y_test,y_pred))
print("R Squared Erro:",metrics.r2_score(y_test,y_pred)) # greater the value of R the most the model best fits.


"""# Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

rc_model=RandomForestClassifier()

"""
from sklear.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size= 0.5)
rc_model.fit(x_train,y_train)
rc_model.score(x_train,y_train)
y_pred = rc_model.predict(x_test)


from sklear import metrics
print("Mean square/absolute error:", metrics.mean_square/absolute_error(y_test,y_pred))
print("R Squared Erro:",metrics.r2_score(y_test,y_pred)) # greater the value of R the most the model best fits.


"""# Support Vector (SV) Classifier

from sklearn.svm import SVC

sc_model=SVC()

"""
from sklear.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size= 0.5)
sc_model.fit(x_train,y_train)
sc_model.score(x_train,y_train)
y_pred = sc_model.predict(x_test)


from sklear import metrics
print("Mean square/absolute error:", metrics.mean_square/absolute_error(y_test,y_pred))
print("R Squared Erro:",metrics.r2_score(y_test,y_pred)) # greater the value of R the most the model best fits.


"""# Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier

gc_model= GradientBoostingClassifier()

"""
from sklear.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size= 0.5)
gc_model.fit(x_train,y_train)
gc_model.score(x_train,y_train)
y_pred = gc_model.predict(x_test)


from sklear import metrics
print("Mean square/absolute error:", metrics.mean_square/absolute_error(y_test,y_pred))
print("R Squared Erro:",metrics.r2_score(y_test,y_pred)) # greater the value of R the most the model best fits.


"""# Gradient Boosting Classifier

from xgboost import XGBClassifier

xgb_model= XGBClassifier()

"""
from sklear.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size= 0.5)
xgv_model.fit(x_train,y_train)
xgb_model.score(x_train,y_train)
y_pred = xgb_model.predict(x_test)


from sklear import metrics
print("Mean square/absolute error:", metrics.mean_square/absolute_error(y_test,y_pred))
print("R Squared Erro:",metrics.r2_score(y_test,y_pred)) # greater the value of R the most the model best fits.


"""# KNN Classifier

from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors= 3)

"""
from sklear.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size= 0.5)
neigh.fit(x_train,y_train)
neigh.score(x_train,y_train)
y_pred = neigh.predict(x_test)


from sklear import metrics
print("Mean square/absolute error:", metrics.mean_square/absolute_error(y_test,y_pred))
print("R Squared Erro:",metrics.r2_score(y_test,y_pred)) # greater the value of R the most the model best fits.


"""# ANN Classifier# Naïve Bayes

from sklearn.naive_bayes import GaussianNB, MultinomialNB , BernoulliNB

gnb = GaussianNB()      #   used when data is normally distributed 
mnb = MultinomialNB()   #   used for discrete couns like we have a text problem
bnb = BernoulliNB()     #   used when we have binary data like  0,1
"""
from sklear.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size= 0.5)
gnb.fit(x_train,y_train)
gnb.score(x_train,y_train)
y_pred = gnb.predict(x_test)


from sklear import metrics
print("Mean square/absolute error:", metrics.mean_square/absolute_error(y_test,y_pred))
print("R Squared Erro:",metrics.r2_score(y_test,y_pred)) # greater the value of R the most the model best fits.


"""
# # Clustering!
# K-means clustering

from sklearn.cluster import KMeans

k_means  = KMeans(n_clusters = 3, random_state = 0)

"""
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
    
y_pred = k_means.predict(data)

y_pred = k_means.predict(X)

X['Cluster'] = y_pred
X

df1 = X[X.Cluster ==0]
df2 = X[X.Cluster ==1]          # make that much df's as much clusters created.
df3 = X[X.Cluster ==2]

cent = k_means.cluster_centers_   # centroids
cent

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


"""# Heirarchical Clustering

# for this we have to find the no. of clusters using 'Dendrogram Method'

# Dendromgram Method!
# Dandogram

"""import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(data, method='ward'))
plt.title('DendromGram')
plt.xlabel('No. of clusters', color ='black')
plt.ylabel('Distances', color = 'black')
plt.show()"""

from sklearn.cluster import AgglomerativeClustering

ag_model = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')

"""
ag_model.fit(data)
y_pred = ag_cl.fit_predict(data)
data['clusters'] = y_pred
data.head()

df1 = data[data.clusters==0]
df2 = data[data.clusters==1]
df3 = data[data.clusters==2]



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
"""
# # Association(relationship like bought milk with bread etc)
# Apriori alogrithm

#transictions = []
#for i in range(0,len(data)):
#   transictions.append([str(data.values[i,j]) for j in range(0,20)])
    

#Training Apriori Algorithm
from apyori import apriori

#algo = apriori(transictions,min_support = 0.003, min_confidence=0.2, min_lift = 3,min_length = 2)

#visualizing Results

#MB = list(algo)

#Results = [list(MB[i][0]) for i in range(0,len(MB))]