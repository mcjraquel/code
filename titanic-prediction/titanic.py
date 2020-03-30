# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:30:01 2020

@author: Celestine
"""


import os
os.chdir('D:/Codecademy/titanic')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the passenger train and test datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Update sex column to numerical
train['Sex'] = train['Sex'].map({'female': 1, 'male': 0})
test['Sex'] = test['Sex'].map({'female': 1, 'male': 0})

# Fill the nan values in the age column
train['Age'].fillna(train.Age.mean(), inplace = True)
test['Age'].fillna(test.Age.mean(), inplace = True)
train['Age'] = train.Age.apply(lambda x: 1 if (x >= 10) and (x < 35) else 0)
test['Age'] = test.Age.apply(lambda x: 1 if (x >= 10) and (x < 35) else 0)


# Create a first class column
train['FirstClass'] = train.Pclass.apply(lambda x: 1 if x == 1 else 0)
test['FirstClass'] = test.Pclass.apply(lambda x: 1 if x == 1 else 0)

# Create a second class column
train['SecondClass'] = train.Pclass.apply(lambda x: 1 if x == 2 else 0)
test['SecondClass'] = test.Pclass.apply(lambda x: 1 if x == 2 else 0)

# Create a third class column
train['ThirdClass'] = train.Pclass.apply(lambda x: 1 if x == 3 else 0)
test['ThirdClass'] = test.Pclass.apply(lambda x: 1 if x == 3 else 0)

#Embarked columns
train['S'] = train.Embarked.apply(lambda x: 1 if x == 'S' else 0)
test['S'] = test.Embarked.apply(lambda x: 1 if x == 'S' else 0)

train['C'] = train.Embarked.apply(lambda x: 1 if x == 'C' else 0)
test['C'] = test.Embarked.apply(lambda x: 1 if x == 'C' else 0)

# Select the desired features
features_train = train[['Sex', 'Age', 'FirstClass', 'SecondClass', 'ThirdClass','SibSp', 'Parch']]
survival_train = train[['Survived']]

# Perform train, test, split
x_train, x_test, y_train, y_test = train_test_split(features_train, survival_train, test_size = 0.1)

# Scale the feature data so it has mean = 0 and standard deviation = 1
scaler = StandardScaler()
scaler.fit_transform(x_train)
scaler.transform(x_test)

# Create and train the model
logreg = LogisticRegression()
logreg.fit(x_train, y_train)

# Score the model on the train data
score_tr = logreg.score(x_train, y_train)
print(score_tr)

# Score the model on the test data
score_te = logreg.score(x_test, y_test)
print(score_te)

# Analyze the coefficients
print(logreg.coef_)

'''
train_survived = train[train.Survived == 1]
print(train_survived.groupby('Age').Survived.count())
print(train_survived.Age.std())
'''


# Test passenger features
features_test = test[['Sex', 'Age', 'FirstClass', 'SecondClass', 'ThirdClass', 'SibSp', 'Parch']]

# Scale the sample passenger features
scaler.transform(features_test)

# Make survival predictions!
predictions = logreg.predict(features_test)
pred_pro = logreg.predict_proba(features_test)

# Save to CSV
test['Survived'] = predictions
predictions_df = test[['PassengerId', 'Survived']]
predictions_df.to_csv(r'D:\Codecademy\titanic\raquel_titanic_predictions_v4.csv', index = False, header = True)
