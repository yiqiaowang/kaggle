import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
import pickle
import sklearn.metrics as metrics





x = pd.read_csv("./data/train_x.csv", header=None)
print("loaded x")
y = pd.read_csv("./data/train_y.csv", header=None)
print("loaded y")

x = x.as_matrix()#.reshape(-1, 64, 64)
print("x shape: {}".format(x.shape))
y = y.as_matrix().ravel()
print("y shape: {}".format(y.shape))

clf = DecisionTreeClassifier()
clf = clf.fit(x, y)

pickle.dump(clf, open("./models/decision_tree.sav", 'wb'))

print("fit logistic regression")

y_pred = clf.predict(x)
f1 = metrics.f1_score(y_true=y, y_pred=y_pred, average='micro')

print("calculated f1 measure")

print(f1)