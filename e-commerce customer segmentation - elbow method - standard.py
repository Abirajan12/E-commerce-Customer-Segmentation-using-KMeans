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


# In[11]:


data_to_scale = data.iloc[:, 3: ]
std_scaler = StandardScaler()
standard_scaled_data = std_scaler.fit_transform(data_to_scale)
standard_scaled_data


# In[13]:


# Create Dataframe for scaled data
standard_scaled_data_df = pd.DataFrame(standard_scaled_data,columns = data_to_scale.columns)
standard_scaled_data_df


# In[14]:


from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters  =i, init = 'k-means++', random_state = 0,n_init =10)
    kmeans.fit(standard_scaled_data_df)
    wcss.append(kmeans.inertia_)


# In[15]:


wcss


# In[16]:


plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

