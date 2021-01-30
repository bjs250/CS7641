import pandas as pd
from sklearn.preprocessing import LabelEncoder

data_headers = [
"age",
"workclass",
"fnlwgt",
"education",
"education-num",
"marital-status",
"occupation",
"relationship",
"race",
"sex",
"capital-gain",
"capital-loss",
"hours-per-week",
"native-country",
"result"
]
TRAIN_DATA_FILENAME = "data/adult-train.csv"
TEST_DATA_FILENAME = "data/adult-test.csv"

feature_columns = data_headers[0:-1]
label_column = data_headers[-1]

def preprocess():
    df_train = pd.read_csv(TRAIN_DATA_FILENAME, sep=',', names=data_headers)
    df_test = pd.read_csv(TEST_DATA_FILENAME, sep=',', names=data_headers)

    # Fix categorical data
    df_train = df_train.apply(LabelEncoder().fit_transform)
    df_test = df_test.apply(LabelEncoder().fit_transform)

    X_train = df_train[feature_columns]
    y_train = df_train[label_column]
    X_test = df_test[feature_columns]
    y_test = df_test[label_column]

    return (X_train, y_train, X_test, y_test)