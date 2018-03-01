import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
import pickle
import sklearn.metrics as metrics





# x = pd.read_csv("./data/train_x.csv", header=None)
# print("loaded x")
# y = pd.read_csv("./data/train_y.csv", header=None)
# print("loaded y")

# x = x.as_matrix()#.reshape(-1, 64, 64)
# print("x shape: {}".format(x.shape))
# y = y.as_matrix().ravel()
# print("y shape: {}".format(y.shape))

# clf = DecisionTreeClassifier()
# clf = clf.fit(x, y)

# pickle.dump(clf, open("./models/decision_tree.sav", 'wb'))

# print("fit decision tree")

clf = pickle.load(open("./models/decision_tree.sav", 'rb'))
test_x = pd.read_csv("./data/test_x.csv", header=None)
test_x = test_x.as_matrix()

test_y = clf.predict(test_x)
test_y = np.array(test_y, dtype="int")

data = np.array([[i,test_y[i]] for i in range(len(test_y))], dtype='int')

with open("./predicted_data/decision_tree_y.csv", 'wb') as f:
    f.write(b"Id,Label\n")
    np.savetxt(f, data, fmt="%i", delimiter=',')
