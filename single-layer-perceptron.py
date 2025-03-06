import numpy as np
import matplotlib as mt
import pandas as pd

# Learning AND operator with a Single-Layer Perceptron 

### TRAINING DATA ####################################################################

training_data = np.array([[0, 0], 
                          [0, 1], 
                          [1, 0], 
                          [1, 1]]) # Holds many different test inputs

correct_answers = np.array([0, 0, 0, 1]) # Answers


### FUNCTIONS ########################################################################

def sum_function(input, weights):
    """
    Sums inputs multiplied by their weights.

    - input: array of input data. 
    - weight: array of weights (numbers).
    """
    return np.dot(input, weights) # dot product does the same as x1 * w1 + x2 * w2 + x3 * w3 + ...

def step_function(sum):
    """
    Returns 1 if the sum is greater or equal to LIMIT. 0, otherwise.
    It is our used activation function.
    """

    LIMIT = 1
    return 1 if sum >= LIMIT else 0

def readjustWeights(input, weights, output, expected_output):
    """
    Gets an input and readjusts weights accordingly. Returns the readjusted weights array.
    """

    LEARNING_INDICATOR = 0.1
    error = expected_output - output
    weights = weights + LEARNING_INDICATOR * input * error
    return weights

def training(inputs, answers):
    """
    Takes an array of inputs an array of corresponding answers. Returns the knowledge obtained (array of weights)
    """

    weights = np.array([0, 0])
    TOLERANCE = 0.05 # what percentage of errors are accepted
    accuracy = 0
    errorAmount = 1

    # Training, while the accuracy is below what's accepted
    while accuracy < (1 - TOLERANCE):
        index = 0
        errorAmount = 0

        # For every test input
        for input in inputs:
            result = step_function(sum_function(input, weights))
            expected_value = answers[index]

            # If an error is found, count it for accuracy calculation and readjust weights.
            if result != expected_value:
                errorAmount += 1
                weights = readjustWeights(input, weights, result, expected_value)

            index += 1
        accuracy = (len(inputs)-errorAmount)/len(inputs)

    print(f'Algorithm over. Accuracy: {accuracy*100}%')

    return weights

def adjustedFunction(weights, input):
    """
    Given an array of weights, runs the sum and step functions to receive the adequate (predicted) result.
    """
    return step_function(sum_function(input, weights))

### RUNNING ##########################################################################

appropriate_weights = training(training_data, correct_answers) # knowledge obtained

# Using knowledge obtained to check the classification of different inputs

print(adjustedFunction(appropriate_weights, [0, 0]))
print(adjustedFunction(appropriate_weights, [0, 1]))
print(adjustedFunction(appropriate_weights, [1, 0]))
print(adjustedFunction(appropriate_weights, [1, 1]))