#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import joblib

pd.options.display.max_columns = None
pd.options.display.max_rows = None

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme(style = "darkgrid")

# In[2]:


data = pd.read_csv("archive/Training Data.csv")
data.head()

# In[36]:


label_encoder = LabelEncoder()

for col in ['Married/Single','Car_Ownership']:
    data[col] = label_encoder.fit_transform( data[col] )

# In[30]:


onehot_encoder = OneHotEncoder(sparse = False)
data['House_Ownership'] = onehot_encoder.fit_transform(data['House_Ownership'].values.reshape(-1, 1) )

# In[31]:


high_card_features = ['Profession', 'CITY', 'STATE']

count_encoder = ce.CountEncoder()

# Transform the features, rename the columns with the _count suffix, and join to dataframe
count_encoded = count_encoder.fit_transform( data[high_card_features] )
data = data.join(count_encoded.add_suffix("_count"))

# In[32]:


data.head()

# In[33]:


data= data.drop(labels=['Profession', 'CITY', 'STATE', "Id"], axis=1)

# In[34]:


data.head()


x = data.drop("Risk_Flag", axis=1)
y = data["Risk_Flag"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, stratify = y, random_state = 7)
print(x_test.columns)
print(x_test.iloc[2])


from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


# In[38]:

print("Fitting")

rf_clf = RandomForestClassifier(criterion='gini', bootstrap=True, random_state=100)


smote_sampler = SMOTE(random_state=9)

pipeline = Pipeline(steps = [['smote', smote_sampler], ['classifier', rf_clf]])

pipeline.fit(x_train, y_train)

#rf_clf.fit(x_train,y_train)
#y_pred = rf_clf.predict(x_test)

print("Predicting")

#test_pred = rf_clf.predict([[  ]])

#smote_sampler = SMOTE(random_state=9)
#pipeline = Pipeline(steps = [['smote', smote_sampler], ['classifier', rf_clf]])
#pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)

# In[39]:


from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

print("-------------------------TEST SCORES-----------------------") 
print(f"Recall: { round(recall_score(y_test, y_pred)*100, 4) }")
print(f"Precision: { round(precision_score(y_test, y_pred)*100, 4) }")
print(f"F1-Score: { round(f1_score(y_test, y_pred)*100, 4) }")
print(f"Accuracy score: { round(accuracy_score(y_test, y_pred)*100, 4) }")
print(f"AUC Score: { round(roc_auc_score(y_test, y_pred)*100, 4) }")

# In[ ]:

joblib.dump(pipeline, 'random_forest_model.joblib')

print("Saved")

