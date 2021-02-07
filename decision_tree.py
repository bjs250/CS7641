import preprocessing
import utils

import time
import pickle
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
import matplotlib.pyplot as plt

def evaluate(max_depth):
    X_train, y_train, X_test, y_test = preprocessing.preprocess()

    clf = DecisionTreeClassifier(
        criterion="gini", 
        splitter="best",
        max_depth=max_depth
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
    print("Max Depth:", clf.tree_.max_depth)
    return (train_acc, test_acc, clf.tree_.max_depth)

def max_depth_experiment(should_plot = False):
    X_train, y_train, X_test, y_test = preprocessing.preprocess()
    parameters = {'max_depth':range(2, 36)}
    
    clf = GridSearchCV(    
            DecisionTreeClassifier(
            criterion="gini", 
            splitter="best"
        ),
        parameters, 
        n_jobs=4,
        return_train_score=True
    )
    clf.fit(X_train, y_train)

    if should_plot == True:
        X = parameters["max_depth"]
        Y1 = clf.cv_results_['mean_train_score']
        Y2 = clf.cv_results_['mean_test_score']
        plt.plot(X, Y1, label = "train")
        plt.plot(X, Y2, label = "cross validation")
        plt.xlabel("max_depth")
        plt.ylabel("accuracy (%)")
        plt.xticks(X)
        plt.legend()
        plt.title("Tuning max_depth")
        plt.savefig("figures/decision_tree/max_depth" + utils.get_current_time() + ".png")
        plt.show()

    return clf.best_params_["max_depth"]

def get_best_parameters():
    X_train, y_train, X_test, y_test = preprocessing.preprocess()

    parameters = {
        'criterion':["gini","entropy"],
        'splitter':["best","random"],
        'max_depth':[1, 5, 10, 15, 20]
    }      

    dt = DecisionTreeClassifier()

    clf = GridSearchCV(    
        dt,
        parameters,
        cv=5, 
        n_jobs=-1,
        return_train_score=True,
        verbose=10
    )
    clf.fit(X_train, y_train)

    best_params = clf.best_params_
    file_pi = open('params/decision_tree.obj', 'wb') 
    pickle.dump(best_params, file_pi) 

    return best_params

if __name__ == '__main__':
    # max_depth = max_depth_experiment(should_plot=False)
    # evaluate(max_depth=max_depth)
    if False:
        best_params = get_best_parameters()
    else:
        filehandler = open('params/decision_tree.obj', 'rb') 
        best_params = pickle.load(filehandler)
        print(best_params)
