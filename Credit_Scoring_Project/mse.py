# -*- coding: utf-8 -*-
"""MSE.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WpnMyJJSbBQWCQ67WUrJC3UT2AHb6ZMt
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch

credit=pd.read_csv("/content/Credit.csv")

credit.head()

credit.describe()

print(credit.isna().sum())

credit.info()

for n,v in credit.items():
    if v.dtype == "object":
        credit[n] = v.factorize()[0]

credit.info()



Cols = ['Income',	'Limit',	'Rating',	'Cards',	'Age',	'Education',	'Gender',	'Student',	'Married',	'Ethnicity']

X_cols = credit[Cols] # Features
Y_cols = credit.Balance

# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X_cols, Y_cols, test_size=0.25) # 75% training and 25% test

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

regressor = RandomForestRegressor(n_estimators=50, random_state=0)
#regressor = RandomForestClassifier(n_estimators=750).fit(X_train, y_train)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

y_pred

y_test



from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

