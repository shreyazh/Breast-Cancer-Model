#!/usr/bin/env python
# coding: utf-8

# In[30]:


from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import accuracy_score


# In[ ]:


# Load dataset 
data = load_breast_cancer()


# In[36]:


# Organize our data 
label_names = data['target_names'] 
labels = data['target'] 
feature_names = data['feature_names'] 
features = data['data']


# In[37]:


# Look at our data 
print(label_names) 
print('Class label = ',labels[0]) 
print(feature_names) 
print(features[0])


# In[39]:


# Split our data 
train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=42)


# In[40]:


# Initialize our classifier 
gnb = GaussianNB()


# In[43]:


# Train our classifier 
model = gnb.fit(train, train_labels)


# In[ ]:


# Make predictions
preds = gnb.predict(test) 
print(preds)


# In[44]:


# Evaluate accuracy 
print(accuracy_score(test_labels, preds))


# In[ ]:




