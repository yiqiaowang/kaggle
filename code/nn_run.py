import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
from NN import NeuralNetwork as NN
from preprocessing import preprocess, SIZE

base_url = "http://cs.mcgill.ca/~pbelan17/"


NUM_TRAIN = 5000
NUM_TEST = 1000

data = np.loadtxt(base_url + "train_x_30000.csv", delimiter=',')
y = np.loadtxt(base_url + "train_y_30000.csv", delimiter=',')
y = y[:NUM_TRAIN + NUM_TEST]

data = data[:NUM_TRAIN + NUM_TEST]
x = preprocess(data)

train_x = x[:NUM_TRAIN]
train_y = y[:NUM_TRAIN]

test_x = x[-NUM_TEST:]
test_y = y[-NUM_TEST:]



params = {
    'train_inputs': train_x,
    'train_targets': train_y,
    'layer_dimentions': [SIZE**2, 450, 250, 50, 10],
    'learning_rate': 1e-3,
    'iterations': 250
}

nn = NN(params)
nn.train()

prediction = nn.predict(test_x)
accuracy = accuracy_score(test_y, prediction)
precision = precision_score(test_y, prediction, average='micro')
recall = recall_score(test_y, prediction, average='micro')
f1 = f1_score(test_y, prediction, average='micro')
conf_matrix = confusion_matrix(test_y, prediction)
print("accuracy:", accuracy)
print("precision:", precision)
print("recall:", recall)
print("f1:", f1)
print("confusion matrix:\n", conf_matrix)
