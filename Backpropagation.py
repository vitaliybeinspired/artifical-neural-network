"""
Vitaliy Stepanov
Assignment 3 - Neural Network Backpropagation
Machine Learning

This program trains an algorithm to predict the classification of unseen images that are numbers
One image is stored in a file as one row and 784 columns of normalized 0-255 numbers of 0.1 to 1
Another file has the actualy number
The network stores weights from input to hidden layer and weights from hidden layer to output layer
These weights are updated using a learning rate and calculated errors.

After a long time of training the network, the weights are saved and unseen data is used to test the network.
"""

from random import seed
from random import uniform

seed()  # initilizes random number
import numpy as np
import pandas as pd

# parameters that can be modified
EPOCH = 5
LEARNING_RATE = 0.05
NETWORK_HIDDEN = 32
TRAINING = 10000
TESTING = 10000

images_training = pd.read_csv('data/training60000.csv', header=None, nrows=TRAINING).to_numpy()
images_training_labels = pd.read_csv('data/training60000_labels.csv', header=None, nrows=TRAINING).to_numpy()
images_testing = pd.read_csv("data/testing10000.csv", header=None, nrows=TESTING).to_numpy()
images_testing_labels = pd.read_csv("data/testing10000_labels.csv", header=None, nrows=TESTING).to_numpy()


def random_weight():
    return uniform(-0.05, 0.05)  # random floating point value includes low, excludes high


def create_network_topology(num_inputs, num_hidden, num_output):
    neural_net = {}
    hidden_layer = {}  # Inside neural_net that will hold all hidden layer nodes
    output_layer = {}  # Inside neural_net that will hold all output layer nodes

    for i in range(num_hidden):
        node_name = i  # changed this to integer so I can flip the matrix in backpropagation
        weights_list = []
        # Intialize all weights feeding into hidden node to random values (the 1 is for the additional bias weight w0)
        for j in range(num_inputs + 1):
            weights_list.append(random_weight())

        hidden_layer[node_name] = np.array(object=weights_list, dtype=float)

    # Add hidden layer to parent dictionary
    neural_net["hidden"] = hidden_layer

    for i in range(num_output):
        node_name = i
        weights_list = []

        # Intialize all weights feeding into output node to random values (the 1 is for the additional bias weight w0)
        for j in range(num_hidden + 1):
            weights_list.append(random_weight())

        output_layer[node_name] = np.array(object=weights_list, dtype=float)

    # Add output layer to parent dictionary
    neural_net["output"] = output_layer
    return neural_net


network = create_network_topology(784, NETWORK_HIDDEN, 10)


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def propagate_input_forward(image):
    image = np.append(image, 1)  # add bias input

    # propagate from input to hidden layer weights
    output_hidden = []
    for hidden in network['hidden'].keys():
        net = np.dot(network['hidden'][hidden], image)
        out = sigmoid(net)
        output_hidden.append(out)

    output_hidden.append(float(1))  # add bias to hidden layer's output
    output_hidden = np.array(object=output_hidden, dtype=float)

    # propagate from hidden to output weights
    output_values = []
    for output in network['output'].keys():
        net = np.dot(network['output'][output], output_hidden)
        out = sigmoid(net)
        output_values.append(out)

    # returns the input values used, the output of the hidden weights and the output
    return {'input': image,
            'hidden': output_hidden,
            'output': np.array(object=output_values, dtype=float)}


def propagate_errors_backward(layers_output, target):
    output_values = layers_output["output"]
    target10 = []  # need 10 values for calculation since there are 10 outputs
    for i in range(len(output_values)):
        target10.append(0.01)

    target10[target] = 0.99  # switch real target value on

    # get errors from output layer
    errors_output = []
    for i in range(len(output_values)):
        output = output_values[i]
        error = output * (1 - output) * (target10[i] - output)
        errors_output.append(error)

    # get errors from hidden layer
    errors_hidden = []
    output_hidden = layers_output["hidden"]
    weights_output = network["output"]
    for i in range(len(weights_output[0])):
        weights = []
        for j in range(len(weights_output)):
            weights.append(weights_output[j][i])
        hidden = output_hidden[i]
        error = hidden * (1 - hidden) * (np.dot(weights, errors_output))
        errors_hidden.append(error)

    return {"output": np.array(object=errors_output, dtype=float),
            "hidden": np.array(object=errors_hidden, dtype=float)}


def update_weights(errors, layers_output):
    weights_output = network["output"]
    # update hidden to output weights
    for i in range(len(weights_output)):
        for j in range(len(weights_output[0])):
            weights_output[i][j] = weights_output[i][j] + (
                        LEARNING_RATE * errors["output"][i] * layers_output["hidden"][j])

    weights_hidden = network["hidden"]

    # update input to hidden weights (gradient decent)
    for i in range(len(weights_hidden)):
        for j in range(len(weights_hidden[0])):
            weights_hidden[i][j] = weights_hidden[i][j] + (
                        LEARNING_RATE * errors["hidden"][i] * layers_output["input"][j])


def train_neural_network(images_training, images_training_labels):
    for epoch in range(EPOCH):  # number of times to go through the data
        i = 0
        print(epoch)
        for image in iter(images_training):
            # get image, hidden weights, and output weights as a dictionary with nparray values
            outputs = propagate_input_forward(image)
            target_label = images_training_labels.item(i)
            i = i + 1  # next target_label
            # get errors of output label by calculating vs target label
            errors = propagate_errors_backward(outputs, target_label)
            update_weights(errors, outputs)


train_neural_network(images_training, images_training_labels)


def test_neural_network(images_testing, images_testing_labels):
    correct = 0
    incorrect = 0
    i = 0
    for image in iter(images_testing):
        actual_target = images_testing_labels.item(i)
        i = i + 1
        # feed the image throught the network
        network_outputs = propagate_input_forward(image)
        predicted_target = int(network_outputs["output"].argmax(None))  # gets the index of the highest value
        if actual_target == predicted_target:
            correct += 1
        else:
            incorrect += 1

    print("network: ", 784, "inputs, ", NETWORK_HIDDEN, "hidden, ", 10, " outputs")
    print("epochs: ", EPOCH)
    print("training instances: ", len(images_training))
    print("testing instances: ", len(images_testing))
    print("learning rate: ", LEARNING_RATE)
    print("correct: ", correct)
    print("incorrect: ", incorrect)
    print("accuracy: ", (correct / len(images_testing)) * 100, "%")


test_neural_network(images_testing, images_testing_labels)
