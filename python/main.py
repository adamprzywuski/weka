import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

features = pd.read_csv("credit-g.csv")
print('The shape of our features is:', features.shape)
print(features.describe())
features = pd.get_dummies(features)
print(features.iloc[:,5:].head(5))

labels = np.array(features['class_good'])
print(labels)
features = features.drop('class_good', axis = 1)
features = features.drop('class_bad', axis = 1)

feature_list = list(features.columns)
print(feature_list)
features = np.array(features)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(train_features, train_labels);
predictions = rf.predict(test_features)
print(predictions)
errors = abs(predictions - test_labels)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape[np.isfinite(mape)])
print('Accuracy:', round(accuracy, 2), '%.')
