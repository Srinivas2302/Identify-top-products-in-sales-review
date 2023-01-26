#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram,cut_tree
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt


# In[55]:


df = pd.concat(
    map(pd.read_csv, ['jewelry.csv', 'women.csv','shoes.csv']), ignore_index=True)
print(df)


# In[56]:


print("Dataset size:", df.shape)
print("Dataset head \n", df.head())


# In[57]:


print("column name and data types: \n", df.dtypes)


# In[58]:


df=df[['category', 'subcategory', 'name', 'current_price', 'raw_price', 'discount', 'likes_count']]


# In[59]:


print("column name and data types: \n", df.dtypes)


# In[60]:


# checking features
categorical_cols = df.select_dtypes(include='O').keys()
# display variabels
categorical_cols


# In[61]:


from sklearn.compose import make_column_transformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from collections import defaultdict


encoder_dict = defaultdict(LabelEncoder)
labeled_df = df.apply(lambda x: encoder_dict[x.name].fit_transform(x))

print(labeled_df)


# In[62]:


print("column name and data types: \n", labeled_df.dtypes)


# In[63]:


labeled_df.head()


# In[64]:


dist = pdist(labeled_df, 'euclidean')
linkage_matrix = linkage(dist, method = 'complete') 
plt.figure(figsize=(15,7))
dendrogram(linkage_matrix)
plt.show()


# In[65]:


from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward')  
cluster.fit_predict(labeled_df)


# In[66]:


plt.figure(figsize=(10, 7))  
plt.scatter(labeled_df['discount'], labeled_df['likes_count'], c=cluster.labels_, cmap='rainbow') 


# In[67]:


labels = cut_tree(linkage_matrix, n_clusters=6) 
labeled_df['label'] = labels
print("describe cluster labeled 0: \n", labeled_df[labeled_df['label']==0].describe()) 
print("describe cluster labeled 1: \n", labeled_df[labeled_df['label']==1].describe()) 
print("describe cluster labeled 2: \n", labeled_df[labeled_df['label']==2].describe()) 
print("describe cluster labeled 3: \n", labeled_df[labeled_df['label']==3].describe()) 
print("describe cluster labeled 4: \n", labeled_df[labeled_df['label']==4].describe()) 
print("describe cluster labeled 5: \n", labeled_df[labeled_df['label']==5].describe())


# In[ ]:





# In[ ]:





# In[ ]:




