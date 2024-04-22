import math
import numpy as np

from layer import Layer

class OutputLayer(Layer):

    def fit(self, eta, y, z_s):
        # Derivada función de activación
        dz = self.z - np.power(self.z, 2)
        # Delta de u en j
        self.delta = (y - self.z) * dz
        # Ajuste de pesos de u y delta_W de u en ji
        self.W = self.W + eta * np.outer(z_s, self.delta)

    def _activation(self, inputs):
        return self.sigmoid(self._net_input(inputs))

    def _activationAndValues(self, inputs):
        v = self._net_input(inputs)
        return self.W, v, self.sigmoid(v)

    def _activationAndStore(self, inputs):
        self.v = self._net_input(inputs)
        self.z = self.sigmoid(self.v)
        return self.z