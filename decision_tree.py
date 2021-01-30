import preprocessing
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
import matplotlib.pyplot as plt
from datetime import datetime

def main(max_depth):
    X_train, y_train, X_test, y_test = preprocessing.preprocess()

    clf = DecisionTreeClassifier(
        criterion="gini", 
        splitter="best",
        max_depth=max_depth
    )
    clf = clf.fit(X_train,y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    train_acc  = metrics.accuracy_score(y_train, y_pred_train)
    test_acc = metrics.accuracy_score(y_test, y_pred_test)
    return (train_acc, test_acc, clf.tree_.max_depth)

def max_depth_experiment():
    min_max_depth = 2
    max_max_depth = 36
    X = list()
    Y1 = list()
    Y2 = list()
    for max_depth in range(min_max_depth, max_max_depth + 1):
        train_acc, test_acc, depth = main(max_depth = max_depth)
        X.append(depth)
        Y1.append(train_acc)
        Y2.append(test_acc)
        print("Train:", train_acc)
        print("Test:", test_acc)
        print("Depth:", depth)

    plt.plot(X, Y1, label = "train")
    plt.plot(X, Y2, label = "test")
    plt.xlabel("max_depth")
    plt.ylabel("accuracy as %")
    plt.xticks(X)
    plt.legend()
    plt.title("Tuning max_depth")
    plt.savefig("figures/decision_tree/max_depth" + get_current_time() + ".png")
    plt.show()

def get_current_time():
    now = datetime.now()
    return now.strftime("%m-%d-%Y-%H:%M:%S")

if __name__ == '__main__':
    max_depth_experiment()
    # train_acc, test_acc, max_depth = main(max_depth=5)
    # print("Train:", train_acc)
    # print("Test:", test_acc)
    # print("Depth:", max_depth)
