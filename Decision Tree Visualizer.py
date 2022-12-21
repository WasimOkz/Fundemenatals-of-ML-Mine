#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


# In[2]:


iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = DecisionTreeClassifier(max_leaf_nodes=4, random_state=0)
clf.fit(X_train, y_train)


# In[24]:


tree.plot_tree(clf)
sns.set(rc={'figure.figsize':(30,10)})
plt.show()


# In[ ]:




