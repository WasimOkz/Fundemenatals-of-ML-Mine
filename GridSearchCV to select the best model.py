#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv(r"E:\Math\2.Mine_Interest_\Data_Science_Course\Datasets\for supervised and unsupervised\IRIS.csv")
data.head()


# In[3]:


from sklearn.preprocessing import LabelEncoder

lb_en = LabelEncoder()


# In[4]:


data.species= lb_en.fit_transform(data.species)


# In[5]:


X = data.drop(['species'], axis= 1)
y = data.species
display(X,y)


# In[6]:


# we have to choose the best model for this data set


# In[7]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble     import RandomForestClassifier
from sklearn.tree         import DecisionTreeClassifier
from sklearn.ensemble     import GradientBoostingClassifier
from sklearn.svm          import SVC


# In[8]:


lg_model= LogisticRegression()
rc_model= RandomForestClassifier()
dc_model= DecisionTreeClassifier()
gc_model= GradientBoostingClassifier()
sc_model=SVC()


# In[10]:


model_param = {
    
'svm':{
        'model':SVC(gamma= 'auto'),
        'param':{
        'C':[1,10,20]
            ,
        'kernel':['rbf','linear'] }
},
    
'decesion tree':{
        'model': DecisionTreeClassifier(),
        'param':{
        'max_leaf_nodes':[5,10,12]
        }   
},
    
   
'random forest':{
        'model': RandomForestClassifier(),
        'param':{
        'n_estimators':[4,10,12] }
    
},
    
'Gradient Boosting':{
         'model': GradientBoostingClassifier(),
         'param':{
         'n_estimators':[50,70,100]
         }
    
},

'Logistic Regrssion':{
         'model':LogisticRegression(),
         'param':{
         'C':[1]
         }
    
}
      
    
    
}


# In[11]:


from sklearn.model_selection import GridSearchCV


# In[12]:


scores = []

for model_name,mp in model_param.items():
    
    gr_search = GridSearchCV(mp['model'],mp['param'], cv=5, return_train_score=False)
    gr_search.fit(X,y)
    scores.append({
        'model':model_name,
        'best_score': gr_search.best_score_,
        'best_param': gr_search.best_params_        
    })
    


# In[13]:


df = pd.DataFrame(scores,columns=['model','best_score','best_param'])
df


# # Model with high Best Score will be the best model for the Data Set :)
