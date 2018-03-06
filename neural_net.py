import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


BATCH_SIZE = 250
INPUT_DIMENSION = 64*64
HIDDEN_DIMENSION_1 = 100
HIDDEN_DIMENSION_2 = 10
OUT_DIMENSION = 1


x = pd.read_csv("./data/head_x.csv", header=None).as_matrix()
y = pd.read_csv("./data/head_y.csv", header=None).as_matrix()


w1 = np.random.randn(INPUT_DIMENSION, HIDDEN_DIMENSION_1)
w2 = np.random.randn(HIDDEN_DIMENSION_1, HIDDEN_DIMENSION_2)
w3 = np.random.randn(HIDDEN_DIMENSION_2, OUT_DIMENSION)


step_size = 1e-7
steps = 1000

for step in range(steps):
    # Forward pass: compute predicted y
    hidden_layer_1 = x.dot(w1)
    hidden_relu_1 = np.maximum(hidden_layer_1, 0)
    
    predicted_y = hidden_relu.dot(w2)

    # Compute and print loss
    error = np.square(predicted_y - y).sum()
    # print(step, error)

    
    gradient_predicted_y = 2.0 * (predicted_y - y)
    gradient_w2 = hidden_relu.T.dot(gradient_predicted_y)
    gradient_hidden_relu = gradient_predicted_y.dot(w2.T)
    gradient_h = gradient_hidden_relu.copy()

    gradient_h[hidden_layer < 0] = 0
    gradient_w1 = x.T.dot(gradient_h)

    
    w1 -= step_size * gradient_w1
    w2 -= step_size * gradient_w2


print("error =", error)


score = accuracy_score(y, predicted_y)
print("score = ", score)
