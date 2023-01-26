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


# In[11]:


df.groupby('category').count()['subcategory']


# In[12]:


df.groupby('category').count()['subcategory'].plot()


# ## Scaling the dataset

# In[13]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
dataset_scaled = sc.fit_transform(df[['current_price','raw_price','likes_count', 'discount']])

dataset_scaled = pd.DataFrame(dataset_scaled)


# In[14]:


dataset_scaled.info()


# In[15]:


dataset_scaled.corr()


# In[16]:


dataset_scaled.head()


# ## As we can see current_price and raw_price are highly correlated, so we can discard one of them while designing our model.
# ## As they are both showing dependence, we can take either one of them in our model.

# In[17]:


#Converting target variable from categorical to numerical
enc_y = LabelEncoder()
y = enc_y.fit_transform(df[['category']])


# In[18]:


#X_new = dataset_scaled['current_price','likes_count','discount']


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(dataset_scaled, y,test_size=0.2, 
random_state=42)
print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))


# In[20]:


X_train.describe()


# In[21]:


y_train.mean()


# ## Decision Tree Classifier

# In[22]:


# Train a decision tree model for classification
clf_default = DecisionTreeClassifier(random_state=42)
clf_default.fit(X_train, y_train)


# In[23]:


# Evaluate the trained model with the testing data
y_pred = clf_default.predict(X_test)
# The prediction accuracy
accuracy = accuracy_score(y_pred, y_test)
print('The testing accuracy is: %.4f\n' % accuracy)


# ## KNN Classifier model

# In[24]:


# Build a KNN classifier model
clf_knn = KNeighborsClassifier(n_neighbors=1)
# Train the model with the training data
clf_knn.fit(X_train, y_train)


# In[25]:


y_pred = clf_knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy is: %.4f\n" % accuracy)


# In[ ]:


cv_scores = []
cv_scores_std = []
k_range = range(1, 135, 5)
for i in k_range:
 clf = KNeighborsClassifier(n_neighbors = i)
 scores = cross_val_score(clf, X_train, 
y_train, scoring='accuracy', cv=KFold(n_splits=10, 
shuffle=True))
# print(scores)
 cv_scores.append(scores.mean())
 cv_scores_std.append(scores.std())
# Plot the relationship
# plt.figure(figsize=(15,10)) 
plt.errorbar(k_range, cv_scores, yerr=cv_scores_std, 
marker='x', label='Accuracy')
plt.ylim([0.1, 1.1])
plt.xlabel('$K$')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.show()


# In[ ]:



parameter_grid = {'n_neighbors': range(1, 135, 5)}
knn_clf = KNeighborsClassifier()
gs_knn = GridSearchCV(knn_clf, parameter_grid, 
scoring='accuracy', cv=KFold(n_splits=10, shuffle=True))
gs_knn.fit(X_train, y_train)
print('Best K value: ', gs_knn.best_params_['n_neighbors'])
print('The accuracy: %.4f\n' % gs_knn.best_score_)
# Got the statistics

cv_scores_means = gs_knn.cv_results_['mean_test_score']
cv_scores_stds = gs_knn.cv_results_['std_test_score']
# Plot the relationship
plt.figure(figsize=(15,10))
plt.errorbar(k_range, cv_scores_means, yerr=cv_scores_stds, 
marker='o', label='gs_knn Accuracy') # gs_knn
plt.errorbar(k_range, cv_scores, yerr=cv_scores_std, 
marker='x', label='manual Accuracy') # manual
plt.ylim([0.1, 1.1])
plt.xlabel('$K$')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.show()


# In[ ]:




