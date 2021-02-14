import preprocessing
import utils
import learning_curves

import time
import pickle
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split 
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
        f"Learning Curve - decision tree, dataset: {dataset}",
        output=f"figures/decision_tree/learning_curve_dataset{dataset}_",
        cv=5
    )


def evaluate(dataset, best_params):
    print(best_params)
    X_train, y_train, X_test, y_test = preprocessing.preprocess(dataset)

    clf = DecisionTreeClassifier(
        criterion=best_params['criterion'], 
        splitter=best_params['splitter'],
        max_depth=best_params['max_depth'],
        ccp_alpha=0.00025
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

    print(precision_recall_fscore_support(y_test, y_pred_test, average='weighted'))
    print(confusion_matrix(y_test, y_pred_test))
    print(classification_report(y_test, y_pred_test))

def get_best_parameters(dataset):
    X_train, y_train, X_test, y_test = preprocessing.preprocess(dataset)

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

def max_depth_experiment(dataset, best_params, should_plot = False):
    X_train, y_train, X_test, y_test = preprocessing.preprocess(dataset)
    parameters = {'max_depth':range(2, 20)}
    
    clf = GridSearchCV(    
            DecisionTreeClassifier(
            criterion=best_params['criterion'], 
            splitter=best_params['splitter']
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
        plt.title("max_depth exploration, dataset 1")
        plt.savefig("figures/decision_tree/max_depth" + utils.get_current_time() + ".png")
        plt.show()

    return clf.best_params_["max_depth"]

def pruning(dataset, best_params):
    X_train, y_train, X_test, y_test = preprocessing.preprocess(dataset)

    clf = DecisionTreeClassifier(
        criterion=best_params['criterion'], 
        splitter=best_params['splitter'],
        max_depth=best_params['max_depth']
    )

    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(
            random_state=0, 
            criterion=best_params['criterion'], 
            splitter=best_params['splitter'],
            max_depth=best_params['max_depth'],
            ccp_alpha=ccp_alpha
        )
        clf.fit(X_train, y_train)
        clfs.append(clf)
    print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
        clfs[-1].tree_.node_count, ccp_alphas[-1]))

    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training and testing sets")
    ax.plot(ccp_alphas, train_scores, marker='o', label="train",
            drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker='o', label="test",
            drawstyle="steps-post")
    ax.legend()
    plt.show()


if __name__ == '__main__':

    if False:
        get_learning_curves(1)
    if False:
        best_params = get_best_parameters(1)
    filehandler = open('params/decision_tree.obj', 'rb') 
    best_params = pickle.load(filehandler)
    if False:
        max_depth_experiment(1, best_params, True)
    if False:
        pruning(1, best_params)
    if True:
        evaluate(1, best_params)
        