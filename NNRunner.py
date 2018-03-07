import pandas as pd
import numpy as np

from NN import NeuralNetwork as NN


def convert_y(y):
    output = []
    for i in y:
        to_add = np.zeros(10, dtype='int')
        to_add[int(i)] = 1
        output.append(to_add)
    return np.array(output)
        



if __name__ == "__main__":

    params = {
            'train_inputs':  pd.read_csv("./data/head_x.csv", header=None).as_matrix(),
            'train_targets': convert_y(pd.read_csv("./data/head_y.csv", header=None).as_matrix()),
            'layer_dimentions': [64*64, 32*64, 1000, 500, 100, 10],
            'learning_rate': 0.005,
            'iterations':  1000
        }


    nn = NN(params)
    nn.train()
    nn.predict(params['train_inputs'])
