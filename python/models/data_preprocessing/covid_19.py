import pandas as pd
from sklearn.preprocessing import LabelEncoder

from models.MultiColumnLabelEncoder import MultiColumnLabelEncoder
pd.set_option('display.max_rows', None)

def preprocess_covid():
    dataset = pd.read_csv("D:\Programming\Python\weka-python\datasets\covid_19.csv")
    print(dataset.head())
    print(dataset.columns)

    dataset = MultiColumnLabelEncoder(columns=["ObservationDate", "Country/Region", "Last Update"]).fit_transform(dataset)
    dataset["Province/State"] = LabelEncoder().fit_transform(dataset["Province/State"].astype(str))


    features = dataset.iloc[:, 0:7].values
    labels = dataset.iloc[:, 7].values

    return features, labels
