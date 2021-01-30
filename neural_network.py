import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics 

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


if __name__ == '__main__':
    main()
