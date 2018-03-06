import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt


def create_lr():
    clf = LogisticRegression()
    print("model made")

    clf = clf.fit(x_train, y_train)
    print("data fitted")

    predicted_y = clf.predict(x_test)
    print("target predicted")

    return accuracy_score(y_test, predicted_y)


def create_knn():
    clf = KNeighborsClassifier()
    print("model made")

    clf = clf.fit(x_train, y_train)
    print("data fitted")

    predicted_y = clf.predict(x_test)
    print("target predicted")

    return accuracy_score(y_test, predicted_y)


def create_rfc():
    clf = RandomForestClassifier()
    print("model made")

    clf = clf.fit(x_train, y_train)
    print("data fitted")

    predicted_y = clf.predict(x_test)
    print("target predicted")

    return accuracy_score(y_test, predicted_y)


def create_gnb():
    clf = GaussianNB()
    print("model made")

    clf = clf.fit(x_train, y_train)
    print("data fitted")

    predicted_y = clf.predict(x_test)
    print("target predicted")

    return accuracy_score(y_test, predicted_y)


def create_svc():
    clf = SVC(C=3.3)
    print("model made")

    clf = clf.fit(x_train, y_train)
    print("data fitted")

    predicted_y = clf.predict(x_test)
    print("target predicted")

    return accuracy_score(y_test, predicted_y)


def test_all():
    print("accuracy of Logisitic Regression :" + str(create_lr()))
    print("accuracy of K Nearest Neighbors :" + str(create_knn()))
    print("accuracy of SVC :" + str(create_svc()))
    print("accuracy of Gaussian NB :" + str(create_gnb()))
    print("accuracy of Random Forest Classifier :" + str(create_rfc()))


def tune_knn(start, stop, step):
    params = np.arange(start, stop, step)
    accuracies = np.zeros(len(params))

    for counter, param in enumerate(params):
        model = KNeighborsClassifier(n_neighbors=param)
        model.fit(x_train, y_train)
        approx_y = model.predict(x_valid)
        accuracies[counter] = accuracy_score(y_valid, approx_y)
        max_index = find_max(accuracies)
        print("progress: " + str(counter) + " out of " + str(len(params)))
    plt.plot(params, accuracies)
    plt.show()
    return start + max_index * step


def tune_svc(start, stop, step):
    params = np.arange(start, stop, step)
    accuracies = np.zeros(len(params))

    for counter, param in enumerate(params):
        model = SVC(C=param)
        model.fit(x_train, y_train)
        approx_y = model.predict(x_valid)
        accuracies[counter] = accuracy_score(y_valid, approx_y)
        max_index = find_max(accuracies)
        print("progress: " + str(counter) + " out of " + str(len(params)))
    plt.plot(params, accuracies)
    plt.show()
    return start + max_index * step


def find_max(data):
    max_val = 0
    cur_max = 0
    for index, value in enumerate(data):
        if value > max_val:
            max_val = value
            cur_max = index
    return cur_max


x_train = pd.read_csv("./data_subset/xzerm ;'"
                      "train_x_20000.csv", header=None)
y_train = pd.read_csv("./data_subset/train_y_20000.csv", header=None)
x_valid = pd.read_csv("./data_subset/valid_x_10000.csv", header=None)
y_valid = pd.read_csv("./data_subset/valid_y_10000.csv", header=None)
x_test = pd.read_csv("./data_subset/test_x_5000.csv", header=None)
y_test = pd.read_csv("./data_subset/test_y_5000.csv", header=None)
print("data loaded")

x_train = preprocessing.scale(x_train.as_matrix())
y_train = y_train.as_matrix().ravel()
x_valid = preprocessing.scale(x_valid.as_matrix())
y_valid = y_valid.as_matrix().ravel()
x_test = preprocessing.scale(x_test.as_matrix())
y_test = y_test.as_matrix().ravel()
print("data reshaped")

#print(tune_svc(2, 4, 0.1))

print("accuracy of SVC with C = 3.3:" + str(create_svc()))




