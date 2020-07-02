import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def classifier(features, labels, classifier):
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25, random_state=42)
    # classifier = RandomForestClassifier(n_estimators=100)
    begin_time = datetime.datetime.now()
    classifier.fit(train_features, train_labels)
    delta = datetime.datetime.now() - begin_time
    y_pred = classifier.predict(test_features)
    return test_labels, y_pred, delta