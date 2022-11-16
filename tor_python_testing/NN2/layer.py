import numpy as np
from numpy.random import Generator, PCG64
rg = Generator(PCG64(1234))


class Neuron:
    def __init__(self, n_inputs):
        #self.weights = (rg.uniform(0, 2, n_inputs) - 1) * 0.0001
        self.weights = np.random.randn(n_inputs) * 0.0001
        self.bias = 0

    def forward(self, inputs):
        """inputs are outputs of previous layer (of what layer this neuron is in) or activation funciton"""
        #returns sum weights_i * inputs_i + bias
        self.output = np.dot(inputs.T, self.weights) + self.bias


class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.output = np.zeros(n_neurons)

        self.neurons = []
        for i in range(n_neurons):
            self.neurons.append(Neuron(n_inputs))

    def forward(self, inputs):
        for j in range(len(self.neurons)):
            self.neurons[j].forward(inputs)
            self.output[j] = self.neurons[j].output
        







if __name__ == '__main__':
    x = np.linspace(0,10,100)

    layer1 = Layer(100, 10)
    layer1.forward(x)
    #n = Neuron(n_inputs = 100)
    print(layer1.output)
    #print(n.bias)
