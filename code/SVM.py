from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt


# create the model, train the model, predict the target, return the metrics
def create_svc():
    clf = SVC(C=3.3)
    print("model made")

    clf = clf.fit(x_train_large, y_train_large)
    print("data fitted")

    predicted_y = clf.predict(x_test)
    print("target predicted")

    accuracy = accuracy_score(y_test, predicted_y)
    precision = precision_score(y_test, predicted_y, average='macro')
    recall = recall_score(y_test, predicted_y, average='macro')
    f1 = f1_score(y_test, predicted_y, average='macro')
    confusion = confusion_matrix(y_test, predicted_y)

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1, 'confusion matrix': confusion}


# test different values of C for the SVM, print plots of metrics versus C, return metrics
def tune_svc(start, stop, step):

    params = np.arange(start, stop, step)
    accuracies = np.zeros(len(params))
    precisions = np.zeros(len(params))
    recalls = np.zeros(len(params))
    f1_scores = np.zeros(len(params))

    for counter, param in enumerate(params):

        model = SVC(C=param)
        model.fit(x_train, y_train)
        approx_y = model.predict(x_valid)

        accuracies[counter] = accuracy_score(y_valid, approx_y)
        precisions[counter] = precision_score(y_valid, approx_y, average='macro')
        recalls[counter] = recall_score(y_valid, approx_y, average='macro')
        f1_scores[counter] = f1_score(y_valid, approx_y, average='macro')

        print("progress: " + str(counter + 1) + " out of " + str(len(params)))

    print(accuracies)
    print(precisions)
    print(recalls)
    print(f1_scores)

    plt.figure("Accuracy versus C for SVM")
    plt.title("Accuracy versus C for SVM")
    plt.plot(params, accuracies)
    plt.xlabel('C')
    plt.ylabel("Accuracy")

    plt.figure("Precision versus C for SVM")
    plt.title("Precision versus C for SVM")
    plt.plot(params, precisions)
    plt.xlabel('C')
    plt.ylabel("Precision")

    plt.figure("Recall versus C for SVM")
    plt.title("Recall versus C for SVM")
    plt.plot(params, precisions)
    plt.xlabel('C')
    plt.ylabel("Recall")

    plt.figure("F1_measure versus C for SVM")
    plt.title("F1_measure versus C for SVM")
    plt.plot(params, precisions)
    plt.xlabel('C')
    plt.ylabel("F1_measure")

    plt.show()
    return{'Accuracy': accuracies, 'Precision': precisions, 'Recall': recalls, 'F1_score': f1_scores}

# load the data
URL_ENDPOINT = "http://cs.mcgill.ca/~ksinha4/datasets/kaggle/"

x = preprocessing.scale(np.loadtxt(URL_ENDPOINT+"train_x.csv", delimiter=",").astype(np.uint8))
y = np.loadtxt(URL_ENDPOINT+"train_y.csv", delimiter=",").astype(np.uint8)

# create subsets of the data to tune the hyper parameter more quickly and to test the model
# training on 5000 data points and validating on 1000 takes roughly 10 minutes value of C
# x_train_large is used to test the data on more than 5000 data points. 15 000 data points are used
# scaling the data to 0 mean, 1 standard deviation increases the performance of the model

x_train = x[:5000]
y_train = y[:5000]
x_valid = x[5000:6000]
y_valid = y[5000:6000]
x_train_large = x[6000:21000]
y_train_large = y[6000:21000]
x_test = x[21000:22000]
y_test = y[21000:22000]

print("data loaded")

# reshape the data

x_train = x_train.as_matrix()
y_train = y_train.as_matrix().ravel()
x_valid = x_valid.as_matrix()
y_valid = y_valid.as_matrix().ravel()
x_train_large = x_train_large.as_matrix()
y_train_large = y_train_large.ravel()
x_test = x_test.as_matrix()
y_test = y_test.ravel()

print("data reshaped")

# tune the hyper parameter
tune_svc(1, 4, 0.25)

# find the metrics of the model with C = 3.3
print(create_svc())


