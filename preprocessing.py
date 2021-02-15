import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data_headers_1 = [
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

def preprocess(dataset):
    
    if dataset == 1:
        TRAIN_DATA_FILENAME = "data/adult-train.csv"
        TEST_DATA_FILENAME = "data/adult-test.csv"
        feature_columns = data_headers_1[0:-1]
        label_column = data_headers_1[-1]

        df_train = pd.read_csv(TRAIN_DATA_FILENAME, sep=',', names=data_headers_1)
        df_test = pd.read_csv(TEST_DATA_FILENAME, sep=',', names=data_headers_1)

        # Fix categorical data
        df_train = df_train.apply(LabelEncoder().fit_transform)
        df_test = df_test.apply(LabelEncoder().fit_transform)

        X_train = df_train[feature_columns]
        y_train = df_train[label_column]
        X_test = df_test[feature_columns]
        y_test = df_test[label_column]

    if dataset == 2:
        DATA_FILENAME = "data/winequality-red.csv"
        df = pd.read_csv(DATA_FILENAME, sep=';')
        X = df.loc[:, df.columns != "quality"]
        y = df.loc[:, df.columns == "quality"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    return (X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    preprocess(2)