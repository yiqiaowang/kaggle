import pandas as pd
import numpy as np
import sys
import tools
import datetime

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from NN import NeuralNetwork as NN

if __name__ == "__main__":

    # Log results into file
    results = open('results.txt', 'a')
    results.write('CV Results\n\n')

    # Load data
    train_inputs = pd.read_csv("./data/head_x.csv", header=None).as_matrix()
    train_targets = pd.read_csv("./data/head_y.csv", header=None).as_matrix()

    test_inputs = pd.read_csv("./data/tail_x.csv", header=None).as_matrix()
    test_targets = pd.read_csv("./data/tail_y.csv", header=None).as_matrix()
    
    # Specify hyper-parameter space
    layer_size = [4096, 50, 10]
    layers = range(1, 2)
    layer_dimentions = [[64*64] + [x]*y + [10] for x in layer_size for y in layers] 

    #learning_rate = [0.001, 0.005, 0.01]
    learning_rate = [0.01]

    #iterations = [100, 250, 500, 1000]
    iterations = [2000]

    param_list = [
            {
            'train_inputs': train_inputs,
            'train_targets': train_targets,
            'layer_dimentions': l_d,
            'learning_rate': l_r,
            'iterations': itr
            }
            for l_d in layer_dimentions
            for l_r in learning_rate
            for itr in iterations
            ]


    for i in range(len(param_list)):
        params = param_list[i]
        print('{}\tTraining model: {}'.format(datetime.datetime.now().time(), i))
        nn = NN(params)
        nn.train()
        prediction = nn.predict(test_inputs)
        
        error = np.square(test_targets - prediction).sum()
        accuracy = accuracy_score(test_targets, prediction, average='micro')
        recall = recall_score(test_targets, prediction, average='micro')  
        precision = precision_score(test_targets, prediction, average='micro') 
        f1 = f1_score(test_targets, prediction, average='micro')
        print('Error: {}\nAccuracy: {}\nRecall: {}\nPrecision: {}\nF1: {}\n\n'.format(error, accuracy, precision, f1))
        results.write('Error: {}\nAccuracy: {}\nRecall: {}\nPrecision: {}\nF1: {}\n\n'.format(error, accuracy, precision, f1))


    results.close()
    # Load previous weights
    #if len(sys.argv) > 1 and sys.argv[1] == 'load':
    #    prev_weights = tools.load_from_bin('models/nn_weights.p')
    #    nn.load_weights(prev_weights)
    #else:
    #    nn.train()
    #    weights = nn.export_weights()
    #    tools.save_as_bin('models/nn_weights.p', weights)

    # print('Prediction: {}'.format(prediction[:5]))
    # print('Target: {}'.format(test_targets[:5]))

    # Report performance
    #error = np.square(test_targets - prediction).sum()
    #score = accuracy_score(test_targets, prediction)
    #print('Error: {}\nAccuracy: {}'.format(error, score))
