
import numpy
from numpy.core.fromnumeric import ndim
import scipy.special


class neuralNetwork:

    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        # set number of nodes
        # TODO: add matrix size validation
        self.inodes = inputNodes
        self.hnodes = hiddenNodes
        self.onodes = outputNodes
        # set learning rate
        self.lr = learningRate
        # initialize random weights of thw form w_i_j from node i to j
        # initial random wieghts centered around 0 with stdev related to
        # number of incoming node links 1/sqrt(n)
        self.wih = numpy.random.normal(
            0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(
            0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # sigmoid functions
        self.activation_func = lambda x: scipy.special.expit(x)


    # train the network
    def train():
        pass

    # query the network
    def query(self, inputs_list):
        # matrix of combined moderated signals into each hidden layer node
        inputs = numpy.array(inputs_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
		
		# calculate signals emerging from hidden layers
        hidden_outputs = self.activation_func(hidden_inputs)

		# calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)

		# calculate signals coming from the outer layer

        return self.activation_func(final_inputs)


