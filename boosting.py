import preprocessing
import utils

import time
import pickle
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics 
import matplotlib.pyplot as plt

def get_best_parameters():
    X_train, y_train, X_test, y_test = preprocessing.preprocess()

    parameters = {
        'n_estimators':[1, 5, 10, 20, 50],
        'learning_rate':[.1,0.5,1.0,1.5],
        'algorithm':['SAMME','SAMME.R']
    }      

    filehandler = open('params/decision_tree.obj', 'rb') 
    dt_parameters = pickle.load(filehandler)

    dt = DecisionTreeClassifier(
        criterion=dt_parameters["criterion"], 
        splitter=dt_parameters["splitter"],
        max_depth=dt_parameters["max_depth"]
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

if __name__ == '__main__':
    # get_learning_curves()
    if False:
        best_params = get_best_parameters()
    else:
        filehandler = open('params/boosting.obj', 'rb') 
        best_params = pickle.load(filehandler)
        print(best_params)
    # evaluate(best_params)