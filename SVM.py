import preprocessing
import learning_curves

import numpy as np
import time
import pickle

from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics 

def get_learning_curves():
    X_train, y_train, X_test, y_test = preprocessing.preprocess()

    clf = svm.SVC(
        kernel='linear',
        verbose=True
    )

    learning_curves.get_learning_curves(
        X_train,
        y_train,
        clf,
        "Learning Curve - SVM",
        output="figures/SVM/learning_curve",
        cv=5
    )

def evaluate(parameters):
    X_train, y_train, X_test, y_test = preprocessing.preprocess()

    np.random.seed(1)
    num_samples = 500
    X_train_sub = X_train.loc[np.random.choice(X_train.index, num_samples, replace=False)]
    y_train_sub = y_train.iloc[X_train_sub.index]
    X_test_sub = X_test.loc[np.random.choice(X_test.index, num_samples, replace=False)]
    y_test_sub = y_test.iloc[X_test_sub.index]

    start = time.time()
    clf = svm.SVC(
        kernel=parameters["kernel"],
        gamma=parameters["gamma"]
    )
    stop = time.time()
    print(f"Training time: {stop - start}s")

    clf.fit(X_train_sub, y_train_sub.values.ravel())
    predictions = clf.predict(X_test_sub)

    y_pred_train = clf.predict(X_train_sub)
    y_pred_test = clf.predict(X_test_sub)

    train_acc  = metrics.accuracy_score(y_train_sub, y_pred_train)
    test_acc = metrics.accuracy_score(y_test_sub, y_pred_test)

    print("Train:", train_acc)
    print("Test:", test_acc)

    print(confusion_matrix(y_test_sub,predictions))
    print(classification_report(y_test_sub,predictions))

def get_best_parameters():
    X_train, y_train, X_test, y_test = preprocessing.preprocess()

    np.random.seed(1)
    num_samples = 500
    X_train_sub = X_train.loc[np.random.choice(X_train.index, num_samples, replace=False)]
    y_train_sub = y_train.iloc[X_train_sub.index]

    parameters = {
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto'],
    }      

    clf = svm.SVC()

    clf = GridSearchCV(    
        clf,
        parameters,
        cv=5, 
        n_jobs=-1,
        return_train_score=True,
        verbose=10
    )
    clf.fit(X_train_sub, y_train_sub)

    best_params = clf.best_params_
    file_pi = open('params/SVM.obj', 'wb') 
    pickle.dump(best_params, file_pi) 

    return best_params

if __name__ == '__main__':
    if False:
        best_params = get_best_parameters()
    else:
        filehandler = open('params/SVM.obj', 'rb') 
        best_params = pickle.load(filehandler)
        print(best_params)
    evaluate(best_params)

