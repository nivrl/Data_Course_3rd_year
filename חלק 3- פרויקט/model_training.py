#!/usr/bin/env python
# coding: utf-8

# ## ML Part- Data Project
# #### Submitted By:
# - **Niv Harel**: 208665869
# - **Eytan Muzafi**: 209160308
# 
# #### Github: [https://github.com/nivrl/Data_Course_3rd_year.git]

# In[1]:


# Please run this cell:
# !pip install geopy


# #### Imports

# In[10]:


import pandas as pd
from datetime import datetime
import numpy as np
import re
import requests
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle
from geopy.geocoders import Nominatim
from car_data_prep import prepare_data


# In[11]:


df = pd.read_csv("original_dataset_second_part.csv")
df_prepared = prepare_data(df)


# 
# ### ML model 

# In[12]:


# Define features and target
X = df_prepared.drop(columns="Price")
y = df_prepared["Price"]

param_grid = {
    'alpha': [0.0001,0.1, 0.5, 1.0, 0.3],
    'l1_ratio': [0.1, 0.5, 0.7, 0.9, 1.0]
}


# In[13]:


elastic_net = ElasticNet()

# Perform grid search with cross-validation 
grid_search = GridSearchCV(elastic_net, param_grid, cv=10, scoring='neg_mean_squared_error')
grid_search.fit(X, y)

# Get the best model from the grid search
best_model = grid_search.best_estimator_


# In[17]:


pickle.dump(best_model,open("trained_model.pkl","wb"))


# In[ ]:


