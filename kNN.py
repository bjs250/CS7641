import preprocessing
import learning_curves

import numpy as np
import time
import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics 

def get_learning_curves():
    X_train, y_train, X_test, y_test = preprocessing.preprocess()

    clf = KNeighborsClassifier()

    learning_curves.get_learning_curves(
        X_train,
        y_train,
        clf,
        "Learning Curve - kNN",
        output="figures/kNN/learning_curve",
        cv=5
    )

def evaluate(parameters):
    X_train, y_train, X_test, y_test = preprocessing.preprocess()

    kNN = KNeighborsClassifier(
        n_neighbors=parameters['n_neighbors'],
        weights=parameters['weights'],
        algorithm=parameters['algorithm']
    )
    kNN.fit(X_train, y_train.values.ravel())
    predictions = kNN.predict(X_test)

    y_pred_train = kNN.predict(X_train)
    y_pred_test = kNN.predict(X_test)

    train_acc  = metrics.accuracy_score(y_train, y_pred_train)
    test_acc = metrics.accuracy_score(y_test, y_pred_test)

    print("Train:", train_acc)
    print("Test:", test_acc)

    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))

def get_best_parameters():
    X_train, y_train, X_test, y_test = preprocessing.preprocess()

    parameters = {
        'n_neighbors':[20, 25, 30, 35, 40, 45, 50],
        'weights':['uniform','distance'],
        'algorithm':['auto','kd_tree']
    }      

    kNN = KNeighborsClassifier()

    clf = GridSearchCV(    
        kNN,
        parameters,
        cv=5, 
        n_jobs=-1,
        return_train_score=True,
        verbose=10
    )
    clf.fit(X_train, y_train)

    best_params = clf.best_params_
    file_pi = open('params/kNN.obj', 'wb') 
    pickle.dump(best_params, file_pi) 

    return best_params

if __name__ == '__main__':
    # get_learning_curves()
    if True:
        best_params = get_best_parameters()
    else:
        filehandler = open('params/kNN.obj', 'rb') 
        best_params = pickle.load(filehandler)
        print(best_params)
    evaluate(best_params)
