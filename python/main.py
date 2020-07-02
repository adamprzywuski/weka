import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

from models.classifier import classifier
from models.data_preprocessing.german_credit import preprocess_german_credit
from models.data_preprocessing.glass import preprocess_glass
from models.data_preprocessing.covid_19 import preprocess_covid
from models.results import calculate_results

np.seterr(divide='ignore', invalid='ignore')

# preprocess
features_gcd, labels_gcd = preprocess_german_credit()
features_glass, labels_glass = preprocess_glass()
features_covid, labels_covid = preprocess_covid()

# classifiers
rf_classifier = RandomForestClassifier(n_estimators=100)
ab_classifier = AdaBoostClassifier(n_estimators=100)
mlp_classifier = MLPClassifier(random_state=1, max_iter=300)

# ## random forest
test_labels_gcd_rf, y_pred_gcd_rf, delta_gcd_rf = classifier(features_gcd, labels_gcd, rf_classifier)
test_labels_glass_rf, y_pred_glass_rf, delta_glass_rf = classifier(features_glass, labels_glass, rf_classifier)
# test_labels_covid_rf, y_pred_covid_rf, delta_covid_rf = classifier(features_covid, labels_covid, rf_classifier)

## adaboost
test_labels_gcd_ab, y_pred_gcd_ab, delta_gcd_ab = classifier(features_gcd, labels_gcd, ab_classifier)
test_labels_glass_ab, y_pred_glass_ab, delta_glass_ab = classifier(features_glass, labels_glass, ab_classifier)

## multilayer perceptron
test_labels_gcd_mlp, y_pred_gcd_mlp, delta_gcd_mlp = classifier(features_gcd, labels_gcd, mlp_classifier)
test_labels_glass_mlp, y_pred_glass_mlp, delta_glass_mlp = classifier(features_glass, labels_glass, mlp_classifier)


# results
print("\nRandom forest results:\n")
print("\tgerman credit data:\n")
calculate_results(test_labels_gcd_rf, y_pred_gcd_rf, delta_gcd_rf)
print("\n\tglass:\n")
calculate_results(test_labels_glass_rf, y_pred_glass_rf, delta_glass_rf)
# print("\n\tcovid-19:")
# calculate_results(test_labels_covid_rf, y_pred_covid_rf, delta_covid_rf)

print("\nAdaboost:\n")
print("\tgerman credit data:\n")
calculate_results(test_labels_gcd_ab, y_pred_gcd_ab, delta_gcd_ab)
print("\n\tglass:\n")
calculate_results(test_labels_glass_ab, y_pred_glass_ab, delta_glass_ab)

print("\nMultilayer perceptron:\n")
print("\tgerman credit data:\n")
calculate_results(test_labels_gcd_mlp, y_pred_gcd_mlp, delta_gcd_mlp)
print("\n\tglass:\n")
calculate_results(test_labels_glass_mlp, y_pred_glass_mlp, delta_glass_mlp)

