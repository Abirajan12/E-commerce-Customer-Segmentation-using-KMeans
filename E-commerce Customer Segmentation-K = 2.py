#!/usr/bin/env python
# coding: utf-8

# # Abstract

#  A key challenge for e-commerce businesses is to analyze the trend in the market to increase their sales. The trend can be easily observed if the companies can group the customers; based on their activity on the e-commerce site. This grouping can be done by applying different criteria like previous orders, mostly searched brands and so on.

# # Problem Statement

#  Given the e-commerce data, use k-means clustering algorithm to cluster customers with similar interest.

# # Dataset Information

#  The data was collected from a well known e-commerce website over a period of time based on the customer’s search profile.

# # Scope

#  ● Analyzing the existing customer data and getting valuable insights about the purchase pattern
#  ● Data pre-processing including missing value treatment
#  ● Segmenting customer based on the optimum number of clusters (‘k’) with the help of silhouette score

# # Data Definition

# Column Description
# 1. Cust_ID Unique numbering for customers
# 2. Gender Gender of the customer
# 3. Orders Number of orders placed by each customer in the past
# 
# Remaining 35 features (brands) contains the number of times
# customers have searched them

# # Content
#    1. [Import Packages](#a1)
#    
#    2. [Read Data](#a2)
#    
#    3. [Understand and Prepare the Data](#a3) 
#    
#        3.1 [Data Types and Dimensions](#a31) 
#        
#        3.2 [Distribution of Variables](#a32)
#        
#        3.3 [Statistical Summary](#a33)
#        
#        3.4 [Duplicated Value](#a34)
#        
#        3.5 [Missing Data Treatment](#a35)
#        
#        3.6 [Visualization](#a36)

# # 1. <a id = a1> Import Packages<a>

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


# # 2. <a id = a2> Read Data<a>

# In[2]:


data = pd.read_excel(r'C:\Users\navin\OneDrive\Desktop\WorkTree\sample_table\cust_data.xlsx')


# In[3]:


data.head()


# # 3. <a id = a3> Understand and Prepare the Data<a>

# ## 3.1 <a id =a31> Data Types and Dimensions <a>

# In[4]:


data.dtypes


# In[5]:


data.info()


# ## 3.2 <a id = a32> Distribution of Variables<a>

# In[6]:


data.shape


# ## 3.3 <a id = a33> Statistical Summary<a>

# In[7]:


data.describe()


# In[8]:


data.describe().T


# ## 3.4 <a id = a34> Duplicated Value<a>

# In[9]:


data.duplicated().sum()


# ## 3.5 <a id = a35> Missing Data Treatment<a>

# In[10]:


data.isnull().sum()


# In[11]:


data['Gender'].unique()


# In[12]:


data['Gender'].value_counts(dropna = False)


# In[13]:


data['Gender'].mode()


# In[14]:


data['Gender'] = data['Gender'].fillna(data['Gender'].mode()[0])


# In[15]:


data['Gender'].value_counts(dropna = False)


# In[16]:


data.isnull().sum()


# ## 3.6 <a id = a36> Visualization <a>

# #### Distribution of data among Genders

# In[17]:


sns.countplot(data, x ='Gender')


# #### Distubution of data among Genders in a pie chart

# In[18]:


fig1 = px.pie(data, names = 'Gender',  title= 'Gender Distribution')
fig1.show()


# In[19]:


data.head()


# #### Total search made by each customer in the e-commerce platform

# In[20]:


total_search = data.iloc[:,3:].sum(axis = 1)
result_df = data[['Cust_ID','Gender','Orders']].copy()
result_df['total_search'] = total_search
result_df


# #### Top 20 customers with maximum of searches

# In[21]:


top_20_customers = result_df.nlargest(20, 'total_search')
top_20_customers


# In[22]:


fig = px.bar(
    top_20_customers,
    x='Cust_ID',
    y='total_search',
    title='Top 20 Customers with the Most Searches',
    labels={'Cust_ID': 'Customer ID', 'total_search': 'Total Search Count'},
    color_discrete_sequence=['red'],
)
for i, row in top_20_customers.iterrows():
    fig.add_annotation(
        x=row['Cust_ID'],
        y=row['total_search'],
        text=row['Cust_ID'],
        showarrow=False,
        font=dict(size=10),
        textangle=90,
        xanchor='center',
    )
fig.show()


# #### Gender wise total_search count

# In[23]:


gender_wise_search_count_data = result_df.groupby('Gender')['total_search'].sum().reset_index()
gender_wise_search_count_data


# In[24]:


fig = px.bar(
    gender_wise_search_count_data,
    x='Gender',
    y='total_search',
    title='Gender-wise Search Count',
    labels={'Gender': 'Gender', 'total_search': 'Total Search Count'},
)
fig.show()


# #### Brand wise total search count

# In[25]:


data.iloc[:,3:]


# In[26]:


brand_df = data.iloc[:,3:].sum().sort_values(ascending = False).reset_index()
brand_df.columns = ['Brand', 'Total_Search_Count']
brand_df


# #### Top 20 brand names with highest total search count

# In[27]:


top_20_brands = brand_df.head(20)


# In[28]:


fig = px.bar(
    top_20_brands,
    x='Brand',
    y='Total_Search_Count',
    title='Top 20 Brands with Highest Search Counts',
    labels={'Brand': 'Brand', 'Total_Search_Count': 'Total Search Count'},
)
fig.show()


# #### Scaling

# #### Using min max scaler

# In[29]:


data_to_scale = data.iloc[:, 3: ]
minmax_scaler = MinMaxScaler()
minmax_scaled_data = minmax_scaler.fit_transform(data_to_scale)
minmax_scaled_data


# In[30]:


# Create Dataframe for scaled data
minmax_scaled_data_df = pd.DataFrame(minmax_scaled_data,columns = data_to_scale.columns)
minmax_scaled_data_df


# In[31]:


best_score = -1
best_k = -1
best_labels = None

silhouette_scores = []  

for n_clusters in range(2,9):
    # Perform clustering using K-means
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=10)
    labels = kmeans.fit_predict(minmax_scaled_data)

    # Calculate the Silhouette Score
    silhouette_avg = silhouette_score(minmax_scaled_data, labels)
    silhouette_scores.append(silhouette_avg)
    
    # Print the Silhouette Score for each number of clusters
    print(f"Number of Clusters: {n_clusters} - Silhouette Score: {silhouette_avg}")

    # Check if the current Silhouette Score is the best
    if silhouette_avg > best_score:
        best_score = silhouette_avg
        best_k = n_clusters
        best_labels = labels

# Print the best k value
print(f"\nBest k value: {best_k} (Silhouette Score: {best_score})")    


# In[32]:


# Plot the line graph
plt.plot(range(2, 9), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters')

# Highlight the best k value
plt.axvline(x=best_k, color='r', linestyle='--', label=f'Best k: {best_k} (Silhouette Score: {best_score:.2f})')
plt.legend()

# Display the plot
plt.show()


# #### Fitting the Kmeans model

# In[33]:


# Fit the K-Means model to your data
kmeans = KMeans(n_clusters=2, random_state =0, n_init = 10)
kmeans.fit(minmax_scaled_data) 

 # Get the cluster centroids and labels
cluster_centers = kmeans.cluster_centers_ 
cluster_assignments = kmeans.labels_  


# In[34]:


plt.scatter(minmax_scaled_data[:, 0], minmax_scaled_data[:, 1], c=cluster_assignments, cmap='viridis')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=10, c='red', label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('Data Points and Cluster Centroids')
plt.show()


# In[52]:


from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score

silhouette_samples_scores = silhouette_samples(minmax_scaled_data, cluster_assignments)
average_silhouette_score = silhouette_score(minmax_scaled_data, cluster_assignments)

# Create a horizontal bar plot for each data point
y_lower = 10  # Initialize the lower y-coordinate

# Create a subplot
fig, ax1 = plt.subplots(1,1)
fig.set_size_inches(8, 6)


# Get a colormap using matplotlib.colormaps.get_cmap
cmap = plt.get_cmap("Spectral")

# Loop through each cluster to plot the silhouette plot
for i in range(len(np.unique(cluster_assignments))):
    cluster_data = silhouette_samples_scores[cluster_assignments == i]
    cluster_data.sort()

    size_cluster_i = cluster_data.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cmap(float(i) / len(np.unique(cluster_assignments)))

    ax1.fill_betweenx(
        np.arange(y_lower, y_upper),
        0,
        cluster_data,
        facecolor=color,
        edgecolor=color,
        alpha=0.7,
    )

    # Label the silhouette plots with their cluster numbers at the middle
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    # Compute the new y_lower for the next plot
    y_lower = y_upper + 10  # 10 for the 0 samples

ax1.set_xlabel("Silhouette Score")
ax1.set_ylabel("Cluster")

# The vertical line for the average silhouette score of all the values
ax1.axvline(x=average_silhouette_score, color="red", linestyle="--")
ax1.text(0.7, 0.95, f'Average Silhouette Score: {average_silhouette_score:.2f}', transform=ax1.transAxes, fontsize=12, verticalalignment='top')

plt.show()


# #### Cluster Data Analysis

# In[35]:


cluster_assignments


# In[36]:


cluster_analysis_data = minmax_scaled_data_df.copy()
cluster_analysis_data['Cluster labels'] = cluster_assignments
cluster_analysis_data


# In[37]:


final_cluster_data = data.copy()
final_cluster_data['Cluster labels'] = cluster_assignments
final_cluster_data


# ### Analysis of each Clusters

# In[38]:


final_cluster_data['Cluster labels'].unique()


# In[39]:


final_cluster_data['Cluster labels'].value_counts()


# #### Cluster 1

# In[40]:


cluster_1 = final_cluster_data[final_cluster_data['Cluster labels'] == 0]


# In[41]:


cluster_1_df = cluster_1.iloc[:,3:-1].sum().sort_values(ascending = False).reset_index()
cluster_1_df.columns = ['Brand', 'count']
cluster_1_df


# In[42]:


top_10_cluster_1 = cluster_1_df.head(10)


# In[43]:


plt.figure(figsize=(10, 6))
plt.bar(top_10_cluster_1['Brand'], top_10_cluster_1['count'], color='blue')
plt.xlabel('Count')
plt.ylabel('Brand')
plt.title('Top 10 Brands in Cluster 1')
plt.show()


# #### Cluster 1: Hewlett Packard, Wrangler, J.M. Smucker, Burberry, H&M, Juniper, Scabel, Gatorade, Dior, Pop chips

# #### Cluster 2

# In[44]:


cluster_2 = final_cluster_data[final_cluster_data['Cluster labels'] == 1]


# In[45]:


cluster_2_df = cluster_2.iloc[:,3:-1].sum().sort_values(ascending = False).reset_index()
cluster_2_df.columns = ['Brand', 'count']
cluster_2_df


# In[46]:


top_10_cluster_2 = cluster_2_df.head(10)


# In[47]:


plt.figure(figsize=(10, 6))
plt.bar(top_10_cluster_2['Brand'], top_10_cluster_2['count'], color='blue')
plt.xlabel('Count')
plt.ylabel('Brand')
plt.title('Top 10 Brands in Cluster 2')
plt.show()


# #### Cluster 2: J.M. Smucker, Juniper, Burberry, Asics, H&M, Jordan, Huawei, Gatorade, Pop Chips, Samsung
