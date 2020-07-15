#!/usr/bin/env python
# coding: utf-8

# # MALICIOUS URL DETECTION

# ## Importing libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# ## Importing the dataset

# In[2]:


data = pd.read_csv("data.csv")

# Labels
y = data["label"]

# Features
url_list = data["url"]

print(data.head())


# ## Feature Scaling

# In[5]:


# Using Tokenizer
vectorizer = TfidfVectorizer()

# Store vectors into X variable as  XFeatures
X = vectorizer.fit_transform(url_list)
print(X)


# ## Splitting the dataset into the Training set and Test set

# In[6]:


# Split into training and testing dataset 80:20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train)


# In[5]:


print(y_train)


# In[8]:


print(X_test)


# In[10]:


print(y_test)


# ## Training the Logistic Regression model on the Training set

# In[ ]:





# In[10]:


# Model Building using logistic regression

classifier = LogisticRegression(random_state = 42)
classifier.fit(X_train, y_train)


# ## Predicting the Test set results & Making the Confusion Matrix

# In[11]:


y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score 

#Confusion Matrix of the Model

print("Confusion Matrix of the model is: ",confusion_matrix(y_test, y_pred))

# Accuracy of the Model

print("Accuracy of our model is: ",accuracy_score(y_test, y_pred))


# ## Visualising the Training set results

# In[12]:


label=["good","bad"]
index_y = np.arange(len(label))
#sample=np.arange(len(y_train))
index_X=[0,0]
for i in y_train:
    if(i=='good'):
        index_X[0] = index_X[0]+1
    else:
        index_X[1] = index_X[1]+1
#print(sample)
#print(index_y)
#print(index_X)
plt.bar(index_y, index_X)
plt.xlabel('Class', fontsize=10)
plt.ylabel('No of URL', fontsize=10)
plt.xticks(index_y, label, fontsize=10, rotation=15)
plt.title('MALACIOUS URL DETECTION [Training Set]')
plt.show()


# ## Visualising the Test set results

# In[13]:


label=["good","bad"]
index_y = np.arange(len(label))
sample=np.arange(len(y_test))
index_X=[0,0]
for i in y_test:
    if(i=='good'):
        index_X[0] = index_X[0]+1
    else:
        index_X[1] = index_X[1]+1
#print(sample)
#print(index_y)
#print(index_X)
plt.bar(index_y, index_X)
plt.xlabel('Class', fontsize=10)
plt.ylabel('No of URL', fontsize=10)
plt.xticks(index_y, label, fontsize=10, rotation=15)
plt.title('MALACIOUS URL DETECTION [Test Set]')
plt.show()


