#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# In[2]:


df = pd.read_csv("/Data/Data.csv")


# In[3]:


df


# In[4]:


# Removing punctuations
data = df.iloc[:,2:27]
data.replace("[^a-zA-z]", " ", regex = True, inplace = True)
# Renaming Columns
list1 = [i for i in range(25)]
new_index = [str(i) for i in list1]
data.columns = new_index


# In[5]:


data


# In[6]:


# Lower casing data set
for i in new_index:
    data[i] = data[i].str.lower()


# In[7]:


data


# In[8]:


headline = []
for i in data.values:
    temp = " ".join([str(j) for j in i])
    headline.append(temp)


# In[9]:


df_new = [[i,j,k] for i,j,k in zip(df.Date, df.Label, headline)]


# In[10]:


df_new = pd.DataFrame(df_new)
df_new.columns = ['Date', 'Label', 'Headline']

train = df_new[df_new['Date']<'20150101']
test = df_new[df_new['Date']>'20141231']


# In[17]:


cv = CountVectorizer(ngram_range=(2,2))
train_data = cv.fit_transform(train.Headline)
test_data = cv.transform(test.Headline)


# In[12]:


random_classifier = RandomForestClassifier(n_estimators=200, criterion = 'entropy')
random_classifier.fit(train_data, train['Label'])


# In[18]:


prediction = random_classifier.predict(test_data)


# In[19]:


matrix = confusion_matrix(test.Label, prediction)
print(matrix)
score = accuracy_score(test.Label, prediction)
print(score)
report = classification_report(test.Label, prediction)
print(report)


# In[ ]:




