import numpy as np
import matplotlib as mt
import pandas as pd
import random

# Learning XOR operator with a Multi-Layer Perceptron, with 1 hidden layer

### TRAINING DATA ####################################################################

training_data = np.array([[0, 0], 
                          [0, 1], 
                          [1, 0], 
                          [1, 1]]) # Holds many different test inputs

correct_answers = np.array([0, 1, 1, 0]) # Answers


### CLASS ############################################################################

class multilayer_perceptron:
    """
    Multilayer perceptron with one hidden layer.
    """

    hidden_layer_size = 3 # number of neurons in hidden layer
    input_size = 2 # number of values in input
    
    weights=np.array([1, 2])
    """
        Matrix. Each column corresponds to the weights of an input value, to be fed to the first layer.  
        A row corresponds to the values to be used by a specific neuron.

        Example:

        -------------------
        |x1 | x2 | x3 |
        |---|---|---|
        |0.32|0.54|0.35|
        |0.12|0.90|0.65|
        ------------------
    """
    
    output_weights = np.array([])
    """
    Array of weights for the output neuron.
    """
    
    def __init__(self, hidden_layer_size, input_size):
        self.hidden_layer_size = hidden_layer_size
        self.input_size = input_size
        
        # Insert an array with random weights for every input connection to a hidden layer neuron
        for neuronIndex in range(self.hidden_layer_size):

            weights_array = np.array([])
            for input in range(self.input_size):
                weights_array = np.append(weights_array, random.random())
            
            self.weights = np.vstack([self.weights, weights_array])

        self.weights = np.delete(self.weights, 0, axis=0)

        # For connections to the output layer
        self.output_weights = np.append(self.output_weights, [random.random() for x in range(hidden_layer_size)])

    def sum_function(input, weights):
        """
        Sums inputs multiplied by their weights.

        - input: array of input data. 
        - weight: array of weights (numbers).
        """
        return np.dot(input, weights) # dot product does the same as x1 * w1 + x2 * w2 + x3 * w3 + ...

    def sigmoid_function(sum):
        """
        It is our used activation function.
        """
        return 1/(1+(np.exp(-sum)))

    def function(self, input):
        
        # input is an array of values, like [0,1]

        # Feed to each neuron in hidden layer
        results = np.array([])
        for neuronIndex in range(self.hidden_layer_size):
            sum_value = multilayer_perceptron.sum_function(input, self.weights[neuronIndex])
            result = multilayer_perceptron.sigmoid_function(sum_value)
            results = np.append(results, result)

        # Feed results to output neuron
        sum_value = multilayer_perceptron.sum_function(results, self.output_weights)
        final_result = multilayer_perceptron.sigmoid_function(sum_value)

        return final_result
    
    def train(self, training_inputs, answers):
        """
            Trains the perceptron with:
            - training_inputs: array of inputs.
            - answers: array of answers to those inputs.
        """
        pass

ml_perceptron = multilayer_perceptron(3, 2)
print(ml_perceptron.weights)
print(ml_perceptron.output_weights)

print(ml_perceptron.function([0, 1]))