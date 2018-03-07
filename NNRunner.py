import pandas as pd

from NN import NeuralNetwork as NN

if __name__ == "__main__":

    params = {
            'train_inputs':  pd.read_csv("./data/head_x.csv", header=None).as_matrix(),
            'train_targets': pd.read_csv("./data/head_y.csv", header=None).as_matrix(),
            'layer_dimentions': [64*64, 100, 1],
            'learning_rate': 0.003,
            'iterations':  1
        }


    nn = NN(params)
    nn.train()
    nn.predict(params['train_inputs'])
