import pandas as pd

from models.MultiColumnLabelEncoder import MultiColumnLabelEncoder


def preprocess_glass():
    dataset = pd.read_csv("datasets/glass.csv")
    print(dataset.head())
    # print(dataset.columns)

    dataset = MultiColumnLabelEncoder(columns=["Type"]).fit_transform(dataset)

    features = dataset.iloc[:, 0:9].values
    labels = dataset.iloc[:, 9].values

    return features, labels