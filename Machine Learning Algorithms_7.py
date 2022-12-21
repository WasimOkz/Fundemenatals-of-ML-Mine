#!/usr/bin/env python
# coding: utf-8

# # Association Algorithms
# Apriori Algorithm

it has 3 parameters.

i)   support 
ii)  confidence
iii) lift


formulas:

support(I)    = transictions containing I/total transictions  :- I for which item you are looking

confidence = transictions containing I1 and I2/ transictions containing I1

lift       =  confidence(I1,I2)/support(I2)
# In[1]:


get_ipython().run_line_magic('pip', 'install apyori')


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv(r"E:\Math\2.Mine_Interest_\Data_Science_Course\Datasets\groceries - groceries.csv")
data


# In[ ]:


transictions = []
for i in range(0,len(data)):
    transictions.append([str(data.values[i,j]) for j in range(0,20)])


# In[ ]:


#Training Apriori Algorithm
from apyori import apriori

algo = apriori(transictions,min_support = 0.003, min_confidence=0.2, min_lift = 3,min_length = 2)


# In[ ]:


#visualizing Results

MB = list(algo)

Results = [list(MB[i][0]) for i in range(0,len(MB))]

