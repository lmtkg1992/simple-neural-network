from numpy import exp, array, random, dot
import csv
import pandas as pd
import argparse
import sys


class NeuralNetwork():
    LEARNING_RATE = 0.9

    def __init__(self):
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        random.seed(1)

        # We model a single neuron, with 3 input connections and 1 output connection.
        # We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
        # and mean 0.
        self.synaptic_weights = 2 * random.random((3, 1)) - 1
        self.synaptic_bias = 0
        self.range_input = {}
        self.range_output = {}

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in xrange(number_of_training_iterations):
            idx = 0
            for training_set_input in training_set_inputs:
                # Pass the training set through our neural network (a single neuron).
                output = self.think(training_set_input)

                # Calculate the error (The difference between the desired output
                # and the predicted output).
                error = (training_set_outputs[idx] - output) * self.__sigmoid_derivative(output)

                # Multiply the error by the input and again by the gradient of the Sigmoid curve
                # This means less confident weights are adjusted more.
                # This means inputs, which are zero, do not cause changes to the weights.
                adjustment = array([training_set_input]).T * error

                # Adjust the weights.
                self.synaptic_weights += self.LEARNING_RATE * adjustment
                # self.synaptic_bias += self.LEARNING_RATE * error
                idx = idx + 1

    # The neural network thinks.
    def think(self, inputs):
        # Pass inputs through our neural network (our single neuron).
        return self.__sigmoid(dot(inputs, self.synaptic_weights) + self.synaptic_bias)


def is_generate_data():
    parser = argparse.ArgumentParser(description='Generate data set')

    # Optional positional argument
    parser.add_argument('generate', nargs='?', help='An optional integer positional argument')
    args = parser.parse_args()

    if args.generate:
        generate_data_set()
        return True

    return False


# get input training data and normalize
def get_input_training_data(neural_network):
    datafile = open('data_set.csv', 'r')
    datareader = csv.reader(datafile, delimiter='\n')
    next(datareader)  # skip header
    data = []
    for row in datareader:
        row_array = row[0].split(',')
        data.append(row_array[:-1])

    data = normalize_data(neural_network, data)

    return data


# get output training data and normalize
def get_output_training_data(neural_network):
    datafile = open('data_set.csv', 'r')
    datareader = csv.reader(datafile, delimiter='\n')
    next(datareader)  # skip header
    data = []
    for row in datareader:
        row_array = row[0].split(',')
        data.append([row_array[-1]])

    data = normalize_data(neural_network, data, 'output')
    data = [max(sub_array) for sub_array in data]

    return data


def normalize_data(neural_network, data, type='input', is_test=False):
    # normalize data in 0->1
    data_transpose = array(array(array(data).T).astype(float))

    maxValue = [max(sub_array) for sub_array in data_transpose]
    minValue = [min(sub_array) for sub_array in data_transpose]

    if not is_test:
        if type == 'output':
            neural_network.range_output['min_value'] = minValue[0]
            neural_network.range_output['max_value'] = maxValue[0]

        if type == 'input':
            neural_network.range_input['min_value'] = minValue
            neural_network.range_input['max_value'] = maxValue

    i = 0
    for entry_feature in data_transpose:
        j = 0
        for entry_value in entry_feature:
            if type == 'output':
                entry_value_normalize = (entry_value - neural_network.range_output['min_value']) / (
                        neural_network.range_output['max_value'] - neural_network.range_output['min_value'])
            if type == 'input':
                median_value = (neural_network.range_input['max_value'][i] + neural_network.range_input['min_value'][
                    i]) / 2
                entry_value_normalize = (entry_value - median_value) / (
                        neural_network.range_input['max_value'][i] - neural_network.range_input['min_value'][i])
            data_transpose[i][j] = entry_value_normalize
            j += 1
        i += 1

    data = array(data_transpose).T

    return data


# Generate data set with secret function
def generate_data_set():
    # Generate out tu to hop cac input
    max_rage_input = 5

    data_export = []
    data_export.append(['x1', 'x2', 'x3', 'y'])

    for x1 in range(0, max_rage_input):
        for x2 in range(0, max_rage_input):
            for x3 in range(0, max_rage_input):
                y = secret_function(x1, x2, x3)
                data_export.append([x1, x2, x3, y])

    df = pd.DataFrame(data_export)
    df.to_csv("data_set.csv", index=False, header=False)

    return


def secret_function(x1, x2, x3):
    return 5 * x1 + 4 * x2 + 8 * x3


def get_test_data():
    return array([[3, 2, 3]])


if __name__ == "__main__":

    # Intialise a single neuron neural network.
    neural_network = NeuralNetwork()

    if is_generate_data():
        sys.exit()

    print "Random starting synaptic weights: "
    print neural_network.synaptic_weights

    # The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = get_input_training_data(neural_network)  # array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_inputs = array(training_set_inputs)

    # chuyen vi ma tran
    training_set_outputs = get_output_training_data(neural_network)  # array([[0, 1, 1, 0]]).T
    training_set_outputs = array([training_set_outputs]).T

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print "New synaptic weights after training: "
    print neural_network.synaptic_weights

    print "New synaptic bias after training: "
    print neural_network.synaptic_bias

    # print "Range synaptic value "
    # print neural_network.range_output

    # Test the neural network with a new situation.
    print "Considering new situation from test data -> ?: "
    test_data = get_test_data()
    test_data = normalize_data(neural_network, test_data, 'input', True)

    # guess output
    # print neural_network.think(test_data)

    # denormalize output
    print neural_network.think(test_data) * (
            neural_network.range_output['max_value'] - neural_network.range_output['min_value']) + \
          neural_network.range_output['min_value']
