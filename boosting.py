import preprocessing
import utils
import learning_curves

import time
import pickle
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics 
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

def get_learning_curves(dataset):
    X_train, y_train, X_test, y_test = preprocessing.preprocess(dataset)

    clf = DecisionTreeClassifier()

    learning_curves.get_learning_curves(
        X_train,
        y_train,
        clf,
        f"Learning Curve - boosting, dataset: {dataset}",
        output=f"figures/boosting/learning_curve_dataset{dataset}_",
        cv=5
    )


def get_best_parameters(dataset):
    X_train, y_train, X_test, y_test = preprocessing.preprocess(dataset)

    parameters = {
        'n_estimators':[1, 5, 10, 20, 50],
        'learning_rate':[.1,0.5,1.0,1.5],
        'algorithm':['SAMME','SAMME.R'],
    }      

    filehandler = open('params/decision_tree.obj', 'rb') 
    dt_parameters = pickle.load(filehandler)

    dt = DecisionTreeClassifier(
        criterion=dt_parameters["criterion"], 
        splitter=dt_parameters["splitter"],
        max_depth=dt_parameters["max_depth"],
        ccp_alpha=0.00030
    )

    boost = AdaBoostClassifier(dt)

    clf = GridSearchCV(    
        boost,
        parameters,
        cv=5, 
        n_jobs=-1,
        return_train_score=True,
        verbose=10
    )
    clf.fit(X_train, y_train)

    best_params = clf.best_params_
    file_pi = open('params/boosting.obj', 'wb') 
    pickle.dump(best_params, file_pi) 

    return best_params

def experiment(dataset, parameters, name, scale, should_plot = True):
    X_train, y_train, X_test, y_test = preprocessing.preprocess(dataset)

    filehandler = open('params/decision_tree.obj', 'rb') 
    dt_parameters = pickle.load(filehandler)

    dt = DecisionTreeClassifier(
        criterion=dt_parameters["criterion"], 
        splitter=dt_parameters["splitter"],
        max_depth=dt_parameters["max_depth"],
        ccp_alpha=0.0003
    )

    boost = AdaBoostClassifier(dt)
    
    clf = GridSearchCV(
        boost,
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
        plt.title(name + " exploration, dataset 1")
        plt.savefig("figures/boosting/" + name + utils.get_current_time() + ".png")
        plt.show()

    return clf.best_params_[name]

def evaluate(dataset, best_params):
    print(best_params)
    X_train, y_train, X_test, y_test = preprocessing.preprocess(dataset)

    filehandler = open('params/decision_tree.obj', 'rb') 
    dt_parameters = pickle.load(filehandler)

    dt = DecisionTreeClassifier(
        criterion=dt_parameters["criterion"], 
        splitter=dt_parameters["splitter"],
        max_depth=dt_parameters["max_depth"],
        ccp_alpha=0.0003
    )

    clf = AdaBoostClassifier(
        dt, 
        algorithm=best_params["algorithm"],
        n_estimators=best_params["n_estimators"],
        learning_rate=best_params["learning_rate"]
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

if __name__ == '__main__':

    if False:
        get_learning_curves(1)
    if False:
        best_params = get_best_parameters(1)
    filehandler = open('params/boosting.obj', 'rb') 
    best_params = pickle.load(filehandler)
    print(best_params)
    if False:
        parameters = {'learning_rate':[0.01, 0.05, 0.10, 0.50, 1.00]}
        name = "learning_rate"
        scale = 'log'
        experiment(1, parameters, name, scale, True)
    if False:
        parameters = {'n_estimators':[1, 5, 10, 15, 20, 25]}
        name = "n_estimators"
        scale = 'linear'
        experiment(1, parameters, name, scale, True)
    if True:
        evaluate(1, best_params)
