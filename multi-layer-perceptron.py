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
    
    weights=np.array([])
    """
        Matrix. 
        
        Each element corresponds to the weight of the connection between input x<sub>i</sub> and neuron<sub>j</sub>.   

        Example:

        -------------------
        |       | neuron1 | neuron2 | neuron3 |
        |---    |---      |---      |---      |
        |**x1** | 0.32    |0.54     |0.35     |
        |**x2** |0.12     |0.90     |0.65     |
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
        self.weights = np.array([[x for x in range(self.hidden_layer_size)]])
        for inputIndex in range(self.input_size):

            weights_array = np.array([])
            for neuronIndex in range(self.hidden_layer_size):
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

    def function(self, inputs):
        """
        Runs the perceptron (function).

        ## Parameters

        - inputs: array of input arrays.

        ## Returns

        - an array of results, for each input.

        ## Why using a matrix multiplication?

        - As a tool: Matrix multiplication allows you to obtain the dot (scalar) product of different vectors/arrays. The result is a matrix with the scalar product of each line of the first matrix by each column of the second.
        - Motivation: For the sum function used, it is desired to obtain the scalar product of each input by its corresponding weights (for each neuron in the hidden layer).
        - Execution: For such, the inputs can be put in a matrix and the weights in another one. That way, we get the scalar product (which is the result of the sum function) of every input by all the involved weights.
    
        Example:

        ```
        inputs =
        [
            [0, 1],
            [0, 0],
            [1, 1]
        ]


        weights = 
        [
            [w1, w2, w3],
            [w4, w5, w6]
        ]


        resulting_matrix = 
        [
            [dot([0,1], [w1, w4]), dot([0,1], [w2, w5]), dot([0,1], [w3, w6])],
            [dot([0,0], [w1, w4]), dot([0,0], [w2, w5]), dot([0,1], [w3, w6])],
            [dot([1,1], [w1, w4]), dot([1,1], [w2, w5]), dot([1,1], [w3, w6])]
        ]
        ```

        Each line of the resulting matrix is the desired sum result for an input, that is, its dot product with the corresponding weights, for each neuron.  
        Then, it is possible to apply the activation function to each sum value (or to a whole line, with NumPy). Therefore, the result of the hidden layer is obtained, and  
        it is possible to proceed to the next step: passing the data onto the output neuron (performing the sum and running the activation function once more).
        
        With this method, it is possible to avoid the use of for loops.

        """

        # each row is an array of weights that have xi as input. Each column corresponds to the weights associated with one neuron.
        # e.g.: a column holds the weights from multiple inputs that are used by sthe sum function in a specific neuron.

        sum_results = np.matmul(inputs, self.weights) # each row is the sum result for a specific input

        hidden_layer_results = multilayer_perceptron.sigmoid_function(sum_results) # each row is a hidden layer result for a specific input. Each element is the output of a neuron.

        # Feed results to output neuron
        sum_value = multilayer_perceptron.sum_function(hidden_layer_results, self.output_weights) # applying dot product between each row and the array of output weights
        final_result = multilayer_perceptron.sigmoid_function(sum_value) # finding final result (applying activation function for each element [sum] in array)

        return final_result
    
    def train(self, training_inputs, answers):
        """
            Trains the perceptron with:
            - training_inputs: array of inputs.
            - answers: array of answers to those inputs.
        """
        pass


### RUNNING ##########################################################################

ml_perceptron = multilayer_perceptron(3, 2)

print('results: ', ml_perceptron.function([[0, 0], [0,1], [1, 0], [1, 1]]))