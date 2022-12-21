#!/usr/bin/env python
# coding: utf-8

# # Support Vectore Machine Alogirthms

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv(r"E:\Math\2.Mine_Interest_\Data_Science_Course\Datasets\Logistic Regression dataset\heart_2020_cleaned.csv")
data.head()


# In[3]:


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


# In[4]:


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


# In[5]:


data


# In[6]:


X = data.iloc[:,1:-1]
y = data.iloc[:,-1]


# In[7]:


display(X,y)


# In[8]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,y , test_size = 0.2)
display(x_train,y_train)
display(x_test,y_test)


# In[9]:


from sklearn.svm import SVC

# sc_model = SVC(kernel = 'rbf')


# In[10]:


# sc_model.fit(x_train,y_train)


# In[11]:


X = np.array([1,2,3,4,5,6]).reshape(-1,1)
y = np.array([2,4,9,16,25,36]).reshape(-1,1)

# with this our data is split in b/w -1,1
from sklearn.preprocessing import StandardScaler

sd= StandardScaler()

X = sd.fit_transform(X)
y = sd.fit_transform(y)


# In[12]:


from sklearn.svm import SVR

sr_model  = SVR(kernel ='rbf')


# In[13]:


sr_model.fit(X,y)


# In[14]:


y_pred = sr_model.predict(X)
y_pred


# In[15]:


plt.scatter(X,y,color = 'r' , marker = '+')
plt.plot(X,sr_model.predict(X), color = 'b')

plt.title('X and y graph')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()


# In[16]:


sr_model.predict(np.array([4]).reshape(-1,1))


# In[17]:


sr_model.score(X,y)


# # KNN Classifier

# In[18]:


data


# In[19]:


X = data.iloc[:,1:]
y = data.iloc[:,0]
display(X,y)


# In[20]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test  = train_test_split(X,y, test_size = 0.2)


# In[21]:


display(x_train,y_train)
display(x_test,y_test)


# In[22]:


from sklearn.neighbors import KNeighborsClassifier

neigh_model = KNeighborsClassifier(n_neighbors = 3)


# In[ ]:


neigh_model.fit(x_train,y_train)


# In[ ]:


# Predictions

y_pred = neigh_model.predict(x_test)
y_pred


# In[ ]:


# checking training score 

neigh_model.score(x_train,y_train)


# In[ ]:


# Testing Score

neigh_model.score(x_test,y_test)


# In[ ]:


# confusion metrics

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))
print('Accurate Values :',confusion_matrix(y_test,y_pred)[0][0],",",confusion_matrix(y_test,y_pred)[1][1])
print('False Values :',confusion_matrix(y_test,y_pred)[0][1],",",confusion_matrix(y_test,y_pred)[1][0])
print('Total Values :',confusion_matrix(y_test,y_pred)[0][0] + confusion_matrix(y_test,y_pred)[0][1] + confusion_matrix(y_test,y_pred)[1][0] + confusion_matrix(y_test,y_pred)[1][1] )


# # Naive Bayes Classifier

# In[ ]:


# we will use it on the same dataset prepared for KNN
# Gaussian
from sklearn.naive_bayes import GaussianNB

gnb_model = GaussianNB()


# In[ ]:


gnb_model.fit(x_train,y_train)


# In[ ]:


# Accuracy
gnb_model.score(x_train,y_train)


# In[ ]:


gnb_model.score(x_test,y_test)


# In[ ]:


y_pred = gnb_model.predict(x_test)
y_pred


# In[ ]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))
print('Accurate Values :',confusion_matrix(y_test,y_pred)[0][0],",",confusion_matrix(y_test,y_pred)[1][1])
print('False Values :',confusion_matrix(y_test,y_pred)[0][1],",",confusion_matrix(y_test,y_pred)[1][0])
print('Total Values :',confusion_matrix(y_test,y_pred)[0][0] + confusion_matrix(y_test,y_pred)[0][1] + confusion_matrix(y_test,y_pred)[1][0] + confusion_matrix(y_test,y_pred)[1][1] )


# In[ ]:


# Multinomial

from sklearn.naive_bayes import MultinomialNB

mnb_model = MultinomialNB()


# In[ ]:


mnb_model.fit(x_train,y_train)


# In[ ]:


y_pred = mnb_model.predict(x_test)
y_pred


# In[ ]:


# Accuracy

mnb_model.score(x_train,y_train)


# In[ ]:


mnb_model.score(x_test,y_test)


# In[ ]:


# Bernoulli 

from sklearn.naive_bayes import BernoulliNB

bnb_model = BernoulliNB()


# In[ ]:


bnb_model.fit(x_train,y_train)


# In[ ]:


y_pred = bnb_model.predict(x_test)
y_pred


# In[ ]:


# Accuracy

bnb_model.score(x_train,y_train)


# In[ ]:


bnb_model.score(x_test,y_test)


# # XGBoost Classifier

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


data = pd.read_csv(r"E:\Math\2.Mine_Interest_\Data_Science_Course\Datasets\for supervised and unsupervised\IRIS.csv")
data.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder

lb_en = LabelEncoder()


# In[ ]:


data['species'] = lb_en.fit_transform(data.species)


# In[ ]:


X = data.iloc[:,1:]
y = data.species


# In[ ]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,y , test_size = 0.17)


# In[ ]:





# In[ ]:





# In[ ]:




