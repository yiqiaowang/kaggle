import numpy as np
import tools

class Layer():
    """
    Class that represents a neural network layer. 
    In this case, only hidden layers and output layer
    are considered. The input layer does not contain weights
    and is therefore not utilized.

    Responsible for maintaining all internal state.
    """

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_deriv(x):
        sig_x = Layer.sigmoid(x)
        return sig_x * (1 - sig_x)

    def __init__(self, input_dim, output_dim):
        # Note that output_dim is equivalent to the number neurons
        # in the layer.
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
    @staticmethod
    def encode_data(data):
        output = []
        for i in data:
            to_add = np.zeros(10, dtype='int')
            to_add[int(i)] = 1
            output.append(to_add)
        return np.array(output)

    @staticmethod
    def decode_data(data):
        return np.apply_along_axis(np.argmax, 1, data)

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
        self.train_targets = NeuralNetwork.encode_data(params['train_targets'])

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

        return  NeuralNetwork.decode_data(subsequent_inputs)

    def export_weights(self):
        return [layer.weights for layer in self.layers]

    def load_weights(self, saved_weights):
        if len(saved_weights) is not len(self.layers):
            raise ValueError('Size of saved weights and layers do not match up')

        for i in range(len(self.layers)):
            self.layers[i].weights = saved_weights[i] 
