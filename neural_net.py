import numpy as np


BATCH_SIZE = 64
INPUT_DIMENSION = 64*64
HIDDEN_DIMENSION = 100
OUT_DIMENSION = 1


x = np.random.randn(BATCH_SIZE, INPUT_DIMENSION)
y = np.random.randn(BATCH_SIZE, OUT_DIMENSION)

w1 = np.random.randn(INPUT_DIMENSION, HIDDEN_DIMENSION)
w2 = np.random.randn(HIDDEN_DIMENSION, OUT_DIMENSION)


step_size = 1e-8
steps = 10000

for step in range(steps):
    # Forward pass: compute predicted y
    hidden_layer = x.dot(w1)
    # print(hidden_layer)
    hidden_relu = np.maximum(hidden_layer, 0)
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