import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import joblib


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme(style = "darkgrid")

# Load the saved Random Forest Classifier from the file
loaded_rf_classifier = joblib.load('random_forest_model.joblib')

print(loaded_rf_classifier)

# Index(['Id', 'Income', 'Age', 'Experience', 'Married/Single',
#        'House_Ownership', 'Car_Ownership', 'CURRENT_JOB_YRS',
#        'CURRENT_HOUSE_YRS', 'Profession_count', 'CITY_count', 'STATE_count'],
#       dtype='object')
# Fitting


data = {
    'Income': [12753279.0],
    'Age': [133.0],
    'Experience': [12.0],
    'Married/Single': [0.0],
    'House_Ownership': [1.0],
    'Car_Ownership': [1.0],
    'CURRENT_JOB_YRS': [12.0],
    'CURRENT_HOUSE_YRS': [12.0],
    'Profession_count': [4782.0],
    'CITY_count': [919.0],
    'STATE_count': [7524.0]
}

#   curl -X POST -H "Content-Type: application/json" -d '{"Income":12753279.0,"Age":133.0,"Experience":12.0,"Married/Single":0.0,"House_Ownership":1.0,"Car_Ownership":1.0,"CURRENT_JOB_YRS":12.0,"CURRENT_HOUSE_YRS":12.0,"Profession_count":4782.0,"CITY_count":919.0,"STATE_count":7524.0}' http://127.0.0.1:5000/predict

test_value = pd.DataFrame(data)

# Now you can use the loaded model to make predictions on new data
predictions = loaded_rf_classifier.predict_proba(test_value)

# Evaluate the accuracy of the loaded model
print(f"Accuracy: {predictions}")