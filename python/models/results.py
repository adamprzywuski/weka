import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


def calculate_results(test_labels, y_pred, delta):
    print(confusion_matrix(test_labels, y_pred))
    print("\n\n")
    print(classification_report(test_labels, y_pred))
    # print(accuracy_score(test_labels, y_pred))
    #
    errors = abs(y_pred - test_labels)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    print('Root Mean Square Error:', round(np.sqrt((errors ** 2).mean()), 2), 'degrees.')
    print(f'Time taken to build the model: {int(delta.total_seconds() * 1000)} ns')
