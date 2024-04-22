import math
import abc
#import random
import numpy as np

class Layer(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, n_inputs, n_neurons, seed=0):
        np.random.seed(seed)
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.delta = None
        self.z = None
        self.v = None
        #orden de parámetros inverso para obtener traspuesta (batches)
        self.W = 0.10 * np.random.randn(n_inputs, n_neurons)

        self.biases = np.zeros((1, n_neurons))

    def _net_input(self, inputs):
        return np.dot(inputs, self.W) + self.biases

    @abc.abstractmethod
    def _activation(self, inputs):
        pass

    @abc.abstractmethod
    def _activationAndValues(self, inputs):
        pass

    @abc.abstractmethod
    def _activationAndStore(self, inputs):
        pass

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def diffSigmoid(x):
        f_x = Layer.sigmoid(x)
        return f_x * (1 - f_x)

    def get_neurons(self):
        return self.n_neurons

    def get_biases(self):
        return self.biases

    def get_weights(self):
        return self.W