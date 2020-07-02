import pandas as pd

from models.MultiColumnLabelEncoder import MultiColumnLabelEncoder


def preprocess_german_credit():
    dataset = pd.read_csv("datasets/credit-g.csv")
    print(dataset.head())
    # print(dataset.columns)

    dataset = MultiColumnLabelEncoder(columns=["checking_status", "credit_history", "purpose", "savings_status",
                                               "employment", "personal_status", "other_parties", "residence_since",
                                               "property_magnitude", "other_payment_plans", "housing", "job",
                                               "own_telephone", "foreign_worker", "class"]).fit_transform(dataset)
    print(dataset.head())

    features = dataset.iloc[:, 0:20].values
    labels = dataset.iloc[:, 20].values

    return features, labels
