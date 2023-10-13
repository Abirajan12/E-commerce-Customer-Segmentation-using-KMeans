#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.preprocessing import MinMaxScaler


# In[2]:


data = pd.read_excel(r'C:\Users\navin\OneDrive\Desktop\WorkTree\sample_table\cust_data.xlsx')


# In[3]:


data.head()


# In[4]:


data['Gender'].unique()


# In[5]:


data['Gender'].value_counts(dropna = False)


# In[6]:


data['Gender'].mode()


# In[7]:


data['Gender'] = data['Gender'].fillna(data['Gender'].mode()[0])


# In[8]:


data['Gender'].value_counts(dropna = False)


# In[9]:


data.isnull().sum()


# In[10]:


data_to_scale = data.iloc[:, 3: ]
minmax_scaler = MinMaxScaler()
minmax_scaled_data = minmax_scaler.fit_transform(data_to_scale)
minmax_scaled_data


# In[11]:


# Create Dataframe for scaled data
minmax_scaled_data_df = pd.DataFrame(minmax_scaled_data,columns = data_to_scale.columns)
minmax_scaled_data_df


# In[13]:


X = minmax_scaled_data_df.iloc[:,:].values


# In[14]:


X


# In[23]:


from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters  =i, init = 'k-means++', random_state = 0,n_init =10)
    kmeans.fit(minmax_scaled_data_df)
    wcss.append(kmeans.inertia_)


# In[24]:


wcss


# In[22]:


plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

