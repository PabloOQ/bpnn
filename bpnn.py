import numpy

from hiddenlayer import HiddenLayer
from outputlayer import OutputLayer

class BPNN(object):

    def __init__(self, input_size, output_size, hidden_layers=2, hidden_size=None, random_state=0):
        self.input_size = input_size
        if hidden_size == None:
            hidden_size = input_size
        self.hidden_layers = []
        self.hidden_layers.append(HiddenLayer(input_size, hidden_size, random_state))
        for i in range(1, hidden_layers):
            self.hidden_layers.append(HiddenLayer(hidden_size, hidden_size, random_state))
        self.output_layer = OutputLayer(hidden_size, output_size, random_state)

    def fit(self, p_X, p_Y, p_eta=0.01, epochs=2):
        layers = self.hidden_layers + [self.output_layer]
        loss = numpy.empty([epochs * p_Y.shape[0]])
        count = 0
        predictions = numpy.empty([epochs * p_Y.shape[0]])
        for _ in range(epochs):
            for i, y in enumerate(p_Y):
                predictions[count] = self.predictAndStore(p_X[i])
                #Usando la última capa como U
                if False:
                    self.output_layer.fit(p_eta, y, self.hidden_layers[len(self.hidden_layers) - 1].z)

                    for j in range(len(self.hidden_layers) - 1, 0, -1):
                        self.hidden_layers[j].fit(p_eta, self.hidden_layers[j - 1].z, self.output_layer.W,
                                                  self.output_layer.delta)
                    self.hidden_layers[0].fit(p_eta, p_X[i], self.output_layer.W, self.output_layer.delta)
                #Usando la siguiente capa como U
                else:
                    self.output_layer.fit(p_eta, y, self.hidden_layers[len(self.hidden_layers) - 1].z)

                    for j in range(len(layers) - 2, 0, -1):
                        layers[j].fit(p_eta, layers[j-1].z, layers[j+1].W, layers[j+1].delta)
                    layers[0].fit(p_eta, p_X[i], layers[1].W, layers[1].delta)
                loss[count] = self.output_layer.delta[0,0]
                count = count + 1

        return loss, predictions




    def predictAndValue(self, x):
        pass

    def predictAndStore(self, x):
        z = x
        for layer in self.hidden_layers:
            # x = self.hidden_layers[i].get_output(x)
            z = layer._activationAndStore(z)

            # Modificar pesos de la capa. Hay que hacer MATHS y pasar lo ultimo del pdf dentro de fit()
        # Output layer
        return self.output_layer._activationAndStore(z)

    def predict(self, x):
        z = x
        for i,layer in enumerate(self.hidden_layers):
            #x = self.hidden_layers[i].get_output(x)
            z = layer._activation(z)

            #Modificar pesos de la capa. Hay que hacer MATHS y pasar lo ultimo del pdf dentro de fit()
        #Output layer
        return self.output_layer._activation(z)

    '''
    def get_network(self):
        return self.network

    def _net_input(self, p_x):
        return

    def _activation(self, p_net_input):
        return

    def _quantization(self, p_activation):
        return
    '''