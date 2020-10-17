
import numpy

class neuralNetwork:

	def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
		# set number of nodes
		self.inodes = inputNodes
		self.hnodes = hiddenNodes
		self.onodes = outputNodes
		#set learning rate
		self.lr = learningRate

		# initialize random weights of thw form w_i_j from node i to j
		# initial random wieghts centered around 0 with stdev related to
		# number of incoming node links 1/sqrt(n)
		self.wih = (numpy.random.normal(0.0, pow(self.inodes), self.hnodes, self.inodes) - 0.5)
		self.who = (numpy.random.normal(0.0, pow(self.hnodes), self.onodes, self.hnodes) - 0.5)

	# train the network
	def train():
		pass

	# query the network
	def query():
		pass