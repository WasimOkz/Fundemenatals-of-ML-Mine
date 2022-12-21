#!/usr/bin/env python
# coding: utf-8

# # ML TASK 01

# In[1]:


morse = {'A': '.-',     'B': '-...',   'C': '-.-.', 
        'D': '-..',    'E': '.',      'F': '..-.',
        'G': '--.',    'H': '....',   'I': '..',
        'J': '.---',   'K': '-.-',    'L': '.-..',
        'M': '--',     'N': '-.',     'O': '---',
        'P': '.--.',   'Q': '--.-',   'R': '.-.',
         'S': '...',    'T': '-',      'U': '..-',
        'V': '...-',   'W': '.--',    'X': '-..-',
        'Y': '-.--',   'Z': '--..'}


# In[2]:


name = input("Enter you name.{Capital letters}:")
print(len(name))


# In[3]:


def str_to_morse(str):
    for i in range(len(name)):
        print(name[i],"=",morse[name[i]])


# In[4]:


str_to_morse(name)


# # morse to str

# In[5]:


val = [ '.--','.-','...','..','--']


# In[6]:


def morse_to_str(value):
    for i in range(len(val)):
        for key, value in morse.items():
            if val[i] == value:
                print(key)


# In[7]:


morse_to_str(val)


# In[ ]:




