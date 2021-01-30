import preprocessing
import learning_curves
import pickle

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics 
import matplotlib.pyplot as plt

def get_learning_curves():
    X_train, y_train, X_test, y_test = preprocessing.preprocess()

    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)

    learning_curves.get_learning_curves(
        X_train,
        y_train,
        mlp,
        "Learning Curve - NN",
        output="figures/NN/learning_curve"
    )

def main():
    X_train, y_train, X_test, y_test = preprocessing.preprocess()

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
    mlp.fit(X_train, y_train.values.ravel())
    predictions = mlp.predict(X_test)

    y_pred_train = mlp.predict(X_train)
    y_pred_test = mlp.predict(X_test)

    train_acc  = metrics.accuracy_score(y_train, y_pred_train)
    test_acc = metrics.accuracy_score(y_test, y_pred_test)

    print("Train:", train_acc)
    print("Test:", test_acc)

    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))


def get_best_parameters():
    X_train, y_train, X_test, y_test = preprocessing.preprocess()

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    parameters = {
        'hidden_layer_sizes': [(10,10,10)],
        # 'activation': ['tanh', 'relu'],
        # 'solver': ['sgd', 'adam'],
        # 'alpha': [0.0001, 0.05],
        # 'learning_rate': ['constant','adaptive'],
    }      

    mlp = MLPClassifier(max_iter=1000)

    clf = GridSearchCV(    
        mlp,
        parameters,
        cv=5, 
        n_jobs=-1,
        return_train_score=True,
        verbose=10
    )
    clf.fit(X_train, y_train)

    best_params = clf.best_params_
    file_pi = open('params/NN.obj', 'wb') 
    pickle.dump(best_params, file_pi) 

    return best_params

if __name__ == '__main__':
    if False:
        best_params = get_best_parameters()
    else:
        filehandler = open('params/NN.obj', 'rb') 
        best_params = pickle.load(filehandler)
    print(best_params)
    # get_learning_curves();
