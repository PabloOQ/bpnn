import numpy as np

from layer import Layer

class HiddenLayer(Layer):

    def fit(self, eta, z_r, W_u, delta_u):
        self.delta = np.empty([self.n_neurons, 1])
        # Delta de s en j
        for j in range(len(self.delta)):
            self.delta[j] = (delta_u.T * W_u[j, :]).sum(axis=1)
            sig = self.z[0, j] - np.power(self.z[0, j], 2)
            self.delta[j] = self.delta[j] * sig
        # Ajuste de pesos de s y delta_W de s en ji
        self.W = self.W + eta * np.outer(z_r, self.delta)

    def _activation(self, inputs):
        return self.sigmoid(self._net_input(inputs))

    def _activationAndValues(self, inputs):
        pass

    def _activationAndStore(self, inputs):
        self.v = self._net_input(inputs)
        self.z = self.sigmoid(self.v)
        return self.z