import preprocessing
import learning_curves
import utils

import numpy as np
import time
import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics 
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt

def get_learning_curves(dataset):
    X_train, y_train, X_test, y_test = preprocessing.preprocess(dataset)

    clf = KNeighborsClassifier()

    learning_curves.get_learning_curves(
        dataset,
        X_train,
        y_train,
        clf,
        f"Learning Curve - kNN, dataset: {dataset}",
        output=f"figures/kNN/learning_curve_dataset{dataset}_",
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

def get_best_parameters(dataset):
    X_train, y_train, X_test, y_test = preprocessing.preprocess(dataset)

    parameters = {
        'n_neighbors':[10, 15, 20, 25, 30, 35, 40, 45, 50],
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

def experiment(dataset, parameters, name, scale, should_plot = True):
    X_train, y_train, X_test, y_test = preprocessing.preprocess(dataset)

    kNN = KNeighborsClassifier()

    clf = GridSearchCV(
        kNN,
        parameters, 
        n_jobs=4,
        return_train_score=True
    )
    clf.fit(X_train, y_train)

    if should_plot == True:
        X = parameters[name]
        Y1 = clf.cv_results_['mean_train_score']
        Y2 = clf.cv_results_['mean_test_score']
        plt.plot(X, Y1, label = "train")
        plt.plot(X, Y2, label = "cross validation")
        plt.xlabel(name)
        plt.ylabel("accuracy (%)")
        plt.xticks(X)
        plt.xscale(scale)
        plt.grid(b=True, which="major")
        plt.legend()
        plt.title(name + " exploration, dataset " + str(dataset))
        plt.savefig("figures/kNN/" + name + utils.get_current_time() + ".png")
        plt.show()

    return clf.best_params_[name]

def evaluate(dataset, best_params):
    print(best_params)
    X_train, y_train, X_test, y_test = preprocessing.preprocess(dataset)

    clf = KNeighborsClassifier(
        n_neighbors=best_params["n_neighbors"],
        weights=best_params["weights"],
        algorithm=best_params["algorithm"]
    )
    
    start = time.time()
    clf = clf.fit(X_train,y_train)
    stop = time.time()
    print(f"Training time: {stop - start}s")
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    train_acc  = metrics.accuracy_score(y_train, y_pred_train)
    test_acc = metrics.accuracy_score(y_test, y_pred_test)
    print("Train Accuracy: ", train_acc)
    print("Test Accuracy", test_acc)

    print(precision_recall_fscore_support(y_test, y_pred_test, average='weighted'))
    print(confusion_matrix(y_test, y_pred_test))
    print(classification_report(y_test, y_pred_test))

    print(classification_report(y_test, y_pred_test))

if __name__ == '__main__':
    if False:
        get_learning_curves(2)
    if False:
        best_params = get_best_parameters(1)
    filehandler = open('params/kNN.obj', 'rb') 
    best_params = pickle.load(filehandler)
    print(best_params)
    if True:
        name = "n_neighbors"
        parameters = {name:[5, 10, 15, 20, 25, 30, 35, 40]}
        scale = 'linear'
        experiment(2, parameters, name, scale, True)
    if False:
        evaluate(1, best_params)
