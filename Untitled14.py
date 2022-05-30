#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[2]:


dataset = sklearn.datasets.load_breast_cancer()

print(dataset)


# In[3]:


df = pd.DataFrame(dataset.data, columns= dataset.feature_names)

df.head()

df.shape

X = df
Y = dataset.target

print(X)


# In[4]:


print(Y)


# In[7]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

print(X.shape, X_train.shape, X_test.shape)


# In[8]:


print(dataset.data.std())


# In[9]:


scaler = StandardScaler()

scaler.fit(X_train)

X_train_standardized = scaler.transform(X_train)

print(X_train_standardized)


# In[10]:


X_test_standardized = scaler.transform(X_test)

print(X_train_standardized.std())


# In[ ]:




