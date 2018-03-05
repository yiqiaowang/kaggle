import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


def create_model(model_name):
    clf = "{}()".format(model_name)
    clf = exec(clf)
    print(model_name)
    x_train = pd.read_csv("./data/train_x.csv", header=None)
    print("loaded x")
    y_train = pd.read_csv("./data/train_y.csv", header=None)
    print("loaded y")

    x_train = x_train.as_matrix()  # reshape(-1, 64, 64)
    print("x shape: {}".format(x_train.shape))
    y_train = y_train.as_matrix().ravel()
    print("y shape: {}".format(y_train.shape))
    clf = clf.fit(x_train, y_train)
    print("data fitted")
    pickle.dump(clf, "./models/{}.sav".format(), 'wb')
    print("pickled")

model_list = ["KNeighborsClassifier", "RandomForestClassifier", "LogisticRegression", "SVC", "GaussianNB"]

for model in model_list:
    create_model(model)




