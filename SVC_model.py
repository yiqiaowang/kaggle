from sklearn.svm import SVC
import pandas as pd
from sklearn import preprocessing


def make_prediction():
    clf = SVC(C=3.3)
    clf = clf.fit(x_train, y_train)
    print("data fitted")
    predicted_y = clf.predict(x_test)
    df = pd.DataFrame(predicted_y)
    with open("./data/target_SVC.csv", 'a') as f:
        f.write("id, label\n")
        df.to_csv(f, header=None)
    return

x_train = pd.read_csv("./data/train_x.csv", header=None)
y_train = pd.read_csv("./data/train_y.csv", header=None)
x_test = pd.read_csv("./data/test_x.csv", header=None)

x_train = preprocessing.scale(x_train.as_matrix())
y_train = y_train.as_matrix().ravel()
x_test = preprocessing.scale(x_test.as_matrix())
print("data reshaped")