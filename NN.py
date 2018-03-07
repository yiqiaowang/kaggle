import numpy as np

from sklearn.metrics import accuracy_score

class Layer():
    """
    Class that represents a neural network layer. 
    Responsible for maintaining all internal state.
    """

    @staticmethod
    def sigmoid(x):
        # FIXME: Numpy complains of overflow runtime expection in np.exp
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_deriv(x):
        return x * (1 - x)

    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim)
        self.outputs = np.array([])
        self.deltas = np.array([])

    def compute_outputs(self, inputs):
        self.outputs = Layer.sigmoid(inputs.dot(self.weights))
        return self.outputs
    
    def compute_deltas(self, error):
        self.deltas = error * Layer.sigmoid_deriv(self.outputs)
        return self.deltas

    def update_weights(self, inputs, rate):
        self.weights += rate * inputs.T.dot(self.deltas)


class NeuralNetwork():
    """
    Class that implements the feed forward Neural Network.
    Takes in a dictionary that specifies the parameters.
    """

    def __init__(self, params):
        # Initialize the hidden layers
        self.layers = []
        layer_dimention_pairs = zip(params['layer_dimentions'][:-1],
                                    params['layer_dimentions'][1:])

        for pair in layer_dimention_pairs:
            self.layers.append(Layer(*pair))

        # Initialize hyper parameters
        self.iterations = params['iterations']
        self.learning_rate = params['learning_rate']
        
        # Store training data
        self.train_inputs = params['train_inputs']
        self.train_targets = params['train_targets']


    def train(self):
        for iteration in range(self.iterations):
            self.forward(self.train_inputs)
            self.loss(self.train_targets)
            self.back_prop(self.train_inputs)


    def forward(self, inputs):
        # Compute values for initial layer
        subsequent_inputs = self.layers[0].compute_outputs(inputs)

        # Propagate forwards through rest of network
        for layer in self.layers[1:]:
            subsequent_inputs = layer.compute_outputs(subsequent_inputs)


    def loss(self, targets):
        # Compute delta for final layer
        previous_deltas = self.layers[-1].compute_deltas(
                targets - self.layers[-1].outputs # Error in final (output) layer
            )

        # Prepare error for previous layer
        output_errors = previous_deltas.dot(self.layers[-1].weights.T)

        # Propagate backwards through rest of network
        for layer in self.layers[-2::-1]:
            previous_deltas = layer.compute_deltas(output_errors)
            output_errors = previous_deltas.dot(layer.weights.T)


    def back_prop(self, inputs):
        # Compute adjustment for initial layer
        self.layers[0].update_weights(inputs, self.learning_rate)
        subsequent_inputs = self.layers[0].outputs

        # Propagate forwards through rest of network
        for layer in self.layers[1:]:
            layer.update_weights(subsequent_inputs, self.learning_rate)
            subsequent_inputs = layer.outputs


    def predict(self, inputs):
        
        # Compute initial output
        subsequent_inputs = self.layers[0].compute_outputs(inputs)
        
        # Propagate forwards through rest of network
        for layer in self.layers[1:]:
            subsequent_inputs = layer.compute_outputs(subsequent_inputs)

        # Report performance metrics

        # FIXME: Currently rounding to nearest int due to numerical issues.
        #        Note: The conversion from float to int may be neccesary.

        # output = subsequent_inputs.copy()
        # output[output < 0.5] = 0
        # output[output >= 0.5] = 1
        output = subsequent_inputs.clip(0, 1).round().astype(int)
        # output = np.rint(subsequent_inputs).astype(int)
        error = np.square(self.train_targets - subsequent_inputs).sum()
        score = accuracy_score(self.train_targets, output)
        print('Error: {}\nAccuracy: {}'.format(error, score))
