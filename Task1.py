#!/usr/bin/env python
# coding: utf-8

# ##  Importing all required librarries
# 

# In[1]:


import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error
from sklearn import tree, datasets
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import pearsonr
from sklearn.preprocessing import OneHotEncoder


# In[2]:


#cd Downloads


# # Reading and merging the selected categories into one singe dataframe

# In[3]:


df_jewelry = pd.read_csv("jewelry.csv")


# In[4]:


df_women = pd.read_csv("women.csv")


# In[5]:


df_shoes = pd.read_csv("shoes.csv")


# In[6]:


df = pd.concat([df_jewelry, df_women, df_shoes])


# In[7]:


df.head()


# In[8]:


df.info()


# In[9]:


#%matplotlib inline 
import matplotlib.pyplot as plt
df.hist(bins=50, figsize=(20,15))
plt.show()


# In[10]:


df.info()


# ## Scaling the dataset

# In[11]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
dataset_scaled = sc.fit_transform(df[['current_price','raw_price','likes_count', 'discount']])

dataset_scaled = pd.DataFrame(dataset_scaled)


# In[12]:


dataset_scaled.info()


# In[13]:


dataset_scaled.corr()


# In[14]:


dataset_scaled.head()


# ## As we can see current_price and raw_price are highly correlated, so we can discard one of them while designing our model.
# ## As they are both showing dependence, we can take either one of them in our model.

# In[15]:


#Converting target variable from categorical to numerical
enc_y = LabelEncoder()
y = enc_y.fit_transform(df[['category']])


# In[16]:


X_new = dataset_scaled


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X_new, y,test_size=0.2, 
random_state=42)
print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))


# In[18]:


X_train.describe()


# In[ ]:




