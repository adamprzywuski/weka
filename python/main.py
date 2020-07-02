import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from models.classifier import classifier
from models.data_preprocessing.german_credit import preprocess_german_credit
from models.data_preprocessing.glass import preprocess_glass
from models.results import calculate_results

np.seterr(divide='ignore', invalid='ignore')

# preprocess
features_gcd, labels_gcd = preprocess_german_credit()
features_glass, labels_glass = preprocess_glass()

# classify
rf_classifier = RandomForestClassifier(n_estimators=100)
ab_classifier = AdaBoostClassifier(n_estimators=100)

## random forest
test_labels_gcd_rf, y_pred_gcd_rf, delta_gcd_rf =classifier(features_gcd, labels_gcd, rf_classifier)
test_labels_glass_rf, y_pred_glass_rf, delta_glass_rf = classifier(features_glass, labels_glass, ab_classifier)

## adaboost
test_labels_gcd_ab, y_pred_gcd_ab, delta_gcd_ab = classifier(features_gcd, labels_gcd, rf_classifier)
test_labels_glass_ab, y_pred_glass_ab, delta_glass_ab = classifier(features_glass, labels_glass, ab_classifier)


# results
print("Random forest results:\n")
print("\tgerman credit data:\n")
calculate_results(test_labels_gcd_rf, y_pred_gcd_rf, delta_gcd_rf)
print("\n\tglass:\n")
calculate_results(test_labels_glass_rf, y_pred_glass_rf, delta_glass_rf)

print("Adaboost:\n")
print("\tgerman credit data:\n")
calculate_results(test_labels_gcd_ab, y_pred_gcd_ab, delta_gcd_ab)
print("\n\tglass:\n")
calculate_results(test_labels_glass_ab, y_pred_glass_ab, delta_glass_ab)

