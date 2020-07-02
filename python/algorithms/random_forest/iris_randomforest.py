import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("iris.csv")
print(dataset.head())

X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values

labelEncoder = preprocessing.LabelEncoder()
y = labelEncoder.fit_transform(y)
print(y)

train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=0.25, random_state=42)

classifier = RandomForestClassifier(n_estimators=5)
classifier.fit(train_features, train_labels)
y_pred = classifier.predict(test_features)
print(classifier.predict([[6.1,3.0,4.6,1.4]]))
print("----------------------------------")

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(test_labels,y_pred))
print(classification_report(test_labels,y_pred))
print(accuracy_score(test_labels, y_pred))


# # vec = CountVectorizer()
# # y = vec.fit_transform(y)
# # print(y.toarray())
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
#
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
#
# regressor = RandomForestClassifier(n_estimators=20, random_state=0)
# regressor.fit(X_train.argmax(axis=1), y_train.argmax(axis=1))
# y_pred = regressor.predict(X_test)
#
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#
# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))
# print(accuracy_score(y_test, y_pred))