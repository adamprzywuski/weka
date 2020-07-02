import datetime

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('always')

dataset = pd.read_csv("../../datasets/glass.csv")
print(dataset.head())
print(dataset.columns)

X = dataset.iloc[:, 0:9].values
y = dataset.iloc[:, 9].values

labelEncoder = preprocessing.LabelEncoder()
y = labelEncoder.fit_transform(y)
print(y)

train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=0.25, random_state=42)

classifier = RandomForestClassifier(n_estimators=5)
begin_time = datetime.datetime.now()
classifier.fit(train_features, train_labels)
delta = datetime.datetime.now() - begin_time
y_pred = classifier.predict(test_features)
# # print(classifier.predict([[6.1,3.0,4.6,1.4]]))
# print("----------------------------------")
#
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#
print(confusion_matrix(test_labels,y_pred))
print(classification_report(test_labels,y_pred))
# print(accuracy_score(test_labels, y_pred))
#
errors = abs(y_pred - test_labels)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
print('Root Mean Square Error:', round(np.sqrt((errors ** 2).mean()), 2), 'degrees.')
print(f'Time taken to build the model: {int(delta.total_seconds() * 1000)} ns')
