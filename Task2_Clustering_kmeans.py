#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram,cut_tree
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt


# In[2]:


df = pd.concat(
    map(pd.read_csv, ['jewelry.csv', 'women.csv','shoes.csv']), ignore_index=True)
print(df)


# In[3]:


print("Dataset size:", df.shape)
print("Dataset head \n", df.head())


# In[4]:


print("column name and data types: \n", df.dtypes)


# In[5]:


df=df[['category', 'subcategory', 'name', 'current_price', 'raw_price', 'discount', 'likes_count']]


# In[6]:


print("column name and data types: \n", df.dtypes)


# In[7]:


# checking features
categorical_cols = df.select_dtypes(include='O').keys()
# display variabels
categorical_cols


# In[8]:


from sklearn.compose import make_column_transformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from collections import defaultdict


encoder_dict = defaultdict(LabelEncoder)
labeled_df = df.apply(lambda x: encoder_dict[x.name].fit_transform(x))

print(labeled_df)


# In[9]:


inverse_transform_lambda = lambda x: encoder_dict[x.name].inverse_transform(x)
unlabeled_df= labeled_df.apply(inverse_transform_lambda)


# In[10]:


print("column name and data types: \n", labeled_df.dtypes)


# In[11]:


labeled_df.head()


# In[12]:


km = KMeans(n_clusters=6) 
km.fit(labeled_df)


# In[13]:


identified_clusters = km.fit_predict(labeled_df)
identified_clusters


# In[14]:


import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
data_with_clusters = df.copy()
data_with_clusters['Clusters'] = identified_clusters 
plt.scatter(data_with_clusters['name'],data_with_clusters['likes_count'],c=data_with_clusters['Clusters'],cmap='rainbow')
plt.scatter(km.cluster_centers_[:, 2], km.cluster_centers_[:, 6], s = 100, c = 'yellow', label = 'Centroids')


# In[15]:


labeled_df['label'] = km.predict(labeled_df)
print("df with cluster labels: \n", labeled_df)


# In[16]:


df_mean = labeled_df.groupby(['label']).agg('mean') 
print(df_mean)


# In[17]:


labeled_df['cluster_ids'] = km.labels_


# In[18]:



# Overall level summary
df_profile_overall = labeled_df.describe().T

# using mean; use appropriate summarization (median, count, etc.) for each feature
df_profile_overall['Overall Dataset'] = df_profile_overall[['mean']]
df_profile_overall = df_profile_overall[['Overall Dataset']]

# Cluster ID level summary
df_cluster_summary = labeled_df.groupby('cluster_ids').describe().T.reset_index()
df_cluster_summary = df_cluster_summary.rename(columns={'level_0':'column','level_1':'metric'})

# using mean; use appropriate summarization (median, count, etc.) for each feature
df_cluster_summary = df_cluster_summary[df_cluster_summary['metric'] == "mean"]
df_cluster_summary = df_cluster_summary.set_index('column')

# join into single summary dataset
df_profile = df_cluster_summary.join(df_profile_overall)


# In[19]:


df_profile


# In[20]:


df_cluster_summary


# In[21]:


df_profile_overall


# In[22]:


labeled_df.nlargest(10,'likes_count')


# In[23]:


print(df['likes_count'].max())


# In[24]:


df.nlargest(10,'likes_count')


# In[25]:


labeled_df


# In[26]:


df


# In[27]:


labeled_df.nlargest(10,'likes_count')


# In[28]:


unlabeled_df.nlargest(10,'likes_count')


# In[ ]:




