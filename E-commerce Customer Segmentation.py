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
#            
#          3.6.1 [Distribution of data among Genders](#a361)
#          
#          3.6.2 [Distubution of data among Genders in a pie chart](#a362)
#          
#          3.6.3 [Total search made by each customer in the e-commerce platform](#a363)
#          
#          3.6.4 [Top 20 customers with maximum of searches](#a364)
#          
#          3.6.5 [Gender wise total_search count](#a365)
#          
#          3.6.6. [Brand wise total search count](#a366)
#          
#          3.6.7 [Top 20 brand names with highest total search count](#a367)
#          
#          
#    4. [Scaling](#a4)
#    
#    5. [Calculating Silhoutte Score](#a5)
#        
#        5.1 [Silhouette Score Analysis for Optimal Number of Clusters](#a51)
#    
#    6. [K-Means Clustering with 3 Clusters: Cluster Centers and Assignments](#a6)
#    
#    7. [Silhoutte Analysis for Cluster Quality](#a7)
#    
#    8. [Cluster Analysis: Exploring Individual Clusters](#a8)
#    
#        8.1 [Cluster 1 Analysis](#a81)
#        
#        8.2 [Cluster 2 Analysis](#a82)
#        
#        8.3 [Cluster 3 Analysis](#a83)
#        
#    9. [Conclusion](#a9)
#    
#        9.1 [Cluster 1](#a91)
#        
#        9.2 [Cluster 2](#a92)
#        
#        9.3 [Cluster 3](#a93)

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

# ### 3.6.1 <a id =a361>Distribution of data among Genders <a>

# In[17]:


sns.countplot(data, x ='Gender')


# ### 3.6.2 <a id =a362> Distubution of data among Genders in a pie chart <a>

# In[18]:


fig1 = px.pie(data, names = 'Gender',  title= 'Gender Distribution')
fig1.show()


# In[19]:


data.head()


# ### 3.6.3 <a id =a363> Total search made by each customer in the e-commerce platform <a>

# In[20]:


total_search = data.iloc[:,3:].sum(axis = 1)
result_df = data[['Cust_ID','Gender','Orders']].copy()
result_df['total_search'] = total_search
result_df


# ### 3.6.4 <a id =a364> Top 20 customers with maximum of searches <a>

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


# ### 3.6.5 <a id =a365> Gender wise total_search count <a>

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


# ### 3.6.6 <a id =a366> Brand wise total search count <a>

# In[25]:


data.iloc[:,3:]


# In[26]:


brand_df = data.iloc[:,3:].sum().sort_values(ascending = False).reset_index()
brand_df.columns = ['Brand', 'Total_Search_Count']
brand_df


# ### 3.6.7 <a id =a367> Top 20 brand names with highest total search count <a>

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


# # 4. <a id = a4>  Scaling <a>

# # Using min max scaler

# In[29]:


data_to_scale = data.iloc[:, 3: ]
minmax_scaler = MinMaxScaler()
minmax_scaled_data = minmax_scaler.fit_transform(data_to_scale)
minmax_scaled_data


# In[30]:


# Create Dataframe for scaled data
minmax_scaled_data_df = pd.DataFrame(minmax_scaled_data,columns = data_to_scale.columns)
minmax_scaled_data_df


# # 5. <a id = a5>  Silhoutte Score <a>

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


# In[79]:


best_labels


# ## 5.1 <a id = a51> Silhouette Score Analysis for Optimal Number of Clusters<a>

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


# # 6. <a id = a6>  K-Means Clustering with 3 Clusters: Cluster Centers and Assignments <a>

# In[56]:


# Fit the K-Means model to your data
kmeans = KMeans(n_clusters=3, random_state =0, n_init = 10)
kmeans.fit(minmax_scaled_data) 

 # Get the cluster centroids and labels
cluster_centers = kmeans.cluster_centers_ 
cluster_assignments = kmeans.labels_  


# ## Data Points and Cluster Centroids Visualization

# In[57]:


plt.scatter(minmax_scaled_data[:, 0], minmax_scaled_data[:, 1], c=cluster_assignments, cmap='viridis')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=10, c='red', label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('Data Points and Cluster Centroids')
plt.show()


# # 7. <a id = a7> Silhoutte Analysis for Cluster Quality <a>
# ### Silhouette analysis is a method to assess the quality of clusters created by a clustering algorithm. It measures how distinct and well-separated the clusters are. A higher silhouette score indicates better-defined clusters(well-separated), while a lower score(overlap or don't look clear) suggests that the clusters may not be meaningful or well-separated. 
# 
# 

# In[87]:


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


# # 8. <a id = a8> Cluster Analysis: Exploring Individual Clusters <a>

# In[58]:


cluster_assignments


# ### Adding Cluster Labels to scaled Data

# In[59]:


cluster_analysis_data = minmax_scaled_data_df.copy()
cluster_analysis_data['Cluster labels'] = cluster_assignments
cluster_analysis_data


# ### Adding Cluster Labels to Original Data

# In[60]:


final_cluster_data = data.copy()
final_cluster_data['Cluster labels'] = cluster_assignments
final_cluster_data


# ### Analysis of each Clusters

# In[61]:


final_cluster_data['Cluster labels'].unique()


# In[62]:


final_cluster_data['Cluster labels'].value_counts()


# ## 8.1 <a id = a81> Cluster 1 Analysis <a>

# In[63]:


# Filter the data to select all rows belonging to Cluster 1
cluster_1 = final_cluster_data[final_cluster_data['Cluster labels'] == 0]


# In[64]:


# Calculate and display the count of brands in Cluster 1
cluster_1_df = cluster_1.iloc[:,3:-1].sum().sort_values(ascending = False).reset_index()
cluster_1_df.columns = ['Brand', 'count']
cluster_1_df


# ### Top 10 Brands in Cluster 1

# In[65]:


top_10_cluster_1 = cluster_1_df.head(10)


# In[66]:


plt.figure(figsize=(10, 6))
plt.bar(top_10_cluster_1['Brand'], top_10_cluster_1['count'], color='blue')
plt.xlabel('Count')
plt.ylabel('Brand')
plt.title('Top 10 Brands in Cluster 1')
plt.show()


# #### Cluster 1: Hewlett Packard, Wrangler, J.M. Smucker, Burberry, H&M, Juniper, Scabel, Gatorade, Dior, Pop chips

# ## 8.2 <a id = a82> Cluster 2 Analysis <a>

# In[67]:


# Filter the data to select all rows belonging to Cluster 2
cluster_2 = final_cluster_data[final_cluster_data['Cluster labels'] == 1]


# In[68]:


# Calculate and display the count of brands in Cluster 2
cluster_2_df = cluster_2.iloc[:,3:-1].sum().sort_values(ascending = False).reset_index()
cluster_2_df.columns = ['Brand', 'count']
cluster_2_df


# ### Top 10 Brands in Cluster 2

# In[69]:


top_10_cluster_2 = cluster_2_df.head(10)


# In[70]:


plt.figure(figsize=(10, 6))
plt.bar(top_10_cluster_2['Brand'], top_10_cluster_2['count'], color='blue')
plt.xlabel('Count')
plt.ylabel('Brand')
plt.title('Top 10 Brands in Cluster 2')
plt.show()


# #### Cluster 2: J.M. Smucker, Juniper, Burberry, Asics, H&M, Jordan, Huawei, Gatorade, Pop Chips, Samsung

# ## 8.3 <a id = a83> Cluster 3 Analysis <a>

# In[71]:


# Filter the data to select all rows belonging to Cluster 3
cluster_3 = final_cluster_data[final_cluster_data['Cluster labels'] == 2]


# In[72]:


# Calculate and display the count of brands in Cluster 3
cluster_3_df = cluster_3.iloc[:,3:-1].sum().sort_values(ascending = False).reset_index()
cluster_3_df.columns = ['Brand', 'count']
cluster_3_df


# ### Top 10 Brands in Cluster 3

# In[73]:


top_10_cluster_3 = cluster_3_df.head(10)


# In[74]:


plt.figure(figsize=(10, 6))
plt.bar(top_10_cluster_3['Brand'], top_10_cluster_3['count'], color='blue')
plt.xlabel('Count')
plt.ylabel('Brand')
plt.title('Top 10 Brands in Cluster 3')
plt.show()


# #### Cluster 3: Scabal, J. M. Smucker, Burberry, Dior, Tommy Hilfiger, H&M, Juniper, Dairy Queen, Jordan, Huawei

# # 9. <a id = a9> Conclusion <a>

# ## 9.1<a id = a91> Cluster 1 <a>

# 1. This group of customers is primarily interested in brands such as Hewlett Packard, Wrangler, J.M. Smucker, Burberry, H&M, Juniper, Scabal, Gatorade, Dior, and Pop Chips. These brands may share similar customer demographics or preferences. Businesses can target this cluster with marketing strategies that focus on these brands.

# 2. Fashion and Clothing - **6**
#    Food and Beverage - **3**
#    Gadgets  - **1**

# | _Cluster 1_ | | |
# |:--------:|:--------:|:--------:|
# |  **Brand**   |  **Industry**   |  **Specific**   |
# |----------|------------|------------|
# |  Hewlett Packard  |  Gadgets   |  Information Technology  |
# |  Wrangler   |  Fashion and Clothing   |     |
# |  J.M. Smucker  |  Food and Beverage   |     |
# |  Burberry   |  Fashion and Clothing   |     |
# |  H&M       |  Fashion and Clothing   |     |
# |  Juniper   |  Fashion and Clothing   | Indian, Jaipur    |
# |  Scabal   |  Fashion and Clothing   |     |
# |  Gatorade   |  Food and Beverage   |  Sport Themed   |
# |  Dior   |  Fashion and Clothing   |     |
# |  Pop Chips   |  Food and Beverage   |  Also Tobacco   |

# 3. This Cluster has predominantly **Fashion and Clothing** , **Food and Beverage** brands in the Top 10.

# ## 9.2<a id = a92> Cluster 2 <a>

# 1. Customers in this cluster show an affinity for brands like J.M. Smucker, Juniper, Burberry, Asics, H&M, Jordan, Huawei, Gatorade, Pop Chips, and Samsung. This cluster might represent a different set of customer preferences compared to Cluster 1. Tailored marketing campaigns for these specific brands can be effective in engaging this segment.

# 2. Fashion and Clothing - **3**
#    Food and Beverage - **3**
#    Sport apparels - **2**
#    Gadgets  - **2**

# | _Cluster 2_ | | |
# |:--------:|:--------:|:--------:|
# |  **Brand**   |  **Industry**   |  **Specific**   |
# |----------|------------|------------|
# |  J.M. Smucker  |  Food and Beverage   |    |
# |  Juniper   |  Fashion and Clothing   |  Indian, Jaipur   |
# |  Burberry  |  Fashion and Clothing   |     |
# |  Asics   |  Sport apparels   |     |
# |  H&M       |  Fashion and Clothing   |     |
# |  Jordan   |  Sport apparels   |     |
# |  Huawei   |  Gadgets   |   Networking Devices  |
# |  Gatorade   |  Food and Beverage   |  Sport Themed   |
# |  Pop chips   |  Food and Beverage   |     |
# |  Samsung   | Gadgets   |  Electronics   |

# 3. This Cluster has a **mixer of all industry brands** in the Top 10.

# ## 9.3<a id = a93> Cluster 3 <a>

# 1. This cluster exhibits interest in brands such as Scabal, J.M. Smucker, Burberry, Dior, Tommy Hilfiger, H&M, Juniper, Dairy Queen, Jordan, and Huawei. It seems to have some overlap with Cluster 2 in terms of brand preferences. Marketing efforts could be designed to explore the connections between these brands and customer behaviors.

# 2. Fashion and Clothing - **6**
#    Food and Beverage - **1**
#    Sport apparels - **1**
#    Gadgets  - **1**
#    Giftcards/Apparels - **1**

# | _Cluster 3_ | | |
# |:--------:|:--------:|:--------:|
# |  **Brand**   |  **Industry**   |  **Specific**   |
# |----------|------------|------------|
# |  Scabal  |  Fashion and Clothing   |    |
# |  J.M. Smucker   |  Food and Beverage   |     |
# |  Burberry  |  Fashion and Clothing   |     |
# |  Dior   |  Fashion and Clothing   |     |
# |  Tommy Hilfiger      |  Fashion and Clothing   |     |
# |  H&M   |  Fashion and Clothing   |     |
# |  Juniper   |  Fashion and Clothing   |  Indian, Jaipur   |
# |  Dairy Queen   |  Gift Cards & Apparels   |  Restaurants   |
# |  Jordan   |  Sport apparels   |     |
# |  Huawei   |  Gadgets   |  Networking Devices   |

# 3. This Cluster has a **Fashion and Clothing** in the Top 10.

# ## Common brands in all the clusters

# ### J. M . Smucker,	Burberry, 	H&M,	Juniper
# Fashion& Clothing - **3**	Food and Beverage -**1**

# ## Common brands between cluster 1 & 2

# ### J. M . Smucker,	Burberry,	H&M,	Juniper,	Gatorade,	Pop chips
# Fashion& Clothing - **3**	Food and Beverage -**3**

# ## Common brands between cluster 1 & 3

# ### J. M . Smucker,	Burberry,	H&M,	Juniper,	Scabal,	 Dior
# Fashion& Clothing - **5**	Food and Beverage -**1**

# ## Common brands between cluster 2 & 3

# ### J. M . Smucker,	Burberry,	H&M,	Juniper,	Jordan,	 Huawei
# Fashion& Clothing - **3**	Food and Beverage -**1**  Sports Apparels - **1** Gadgets - **1**

# ## Project Insights and Implications

# In summary, clustering analysis has helped us identify distinct groups of customers based on their brand preferences. By understanding these clusters, businesses can customize their marketing strategies to better target and engage with each group, potentially leading to improved customer satisfaction and increased sales. Further exploration and targeted campaigns can help uncover deeper insights into these customer segments.
