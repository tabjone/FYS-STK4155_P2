from layer import Layer, Neuron
from activation_functions import *

"""
Cost function: C= - 0.5*(target - predicted)^2
dC/dtarget = t-p
"""
class NeuralNetwork:
    def __init__(self, n_inputs, n_layers, n_neurons):

        self.layer1 = Layer(n_inputs, n_neurons)
        self.activation1 = ActivationSigmoid()

        self.layer2 = Layer(len(self.layer1.neurons), n_neurons)
        self.activation2 = ActivationSigmoid()

        self.layer3 = Layer(len(self.layer2.neurons), n_inputs)
        self.activation3 = ActivationSigmoid()
        
    def forward(self, inputs):
        self.layer1.forward(inputs)
        act1 = self.activation1.forward(self.layer1.output)
        self.layer2.forward(act1)
        act2 = self.activation2.forward(self.layer2.output)
        self.layer3.forward(act2)

        self.output = self.activation3.forward(self.layer3.output)
    
        
    def backprop(self, target, data_input):
        eta = 0.001

        #error layer3
        #now we calculate this for each neuron
        error_output = np.zeros(len(self.layer3.neurons))
        for j in range(len(error_output)):
            error_output[j] = self.activation3.derivative(self.layer3.output[j]) * (target[j] - self.output[j])

        error_layer2 = np.zeros(len(self.layer2.neurons))
        for j in range(len(self.layer2.neurons)):
            error_layer2[j] = 0
            for k in range(len(error_output)):
                error_layer2[j] += error_output[k] * self.layer3.neurons[k].weights[j]
            error_layer2[j] *= self.activation2.derivative(self.layer2.output[j])
        
        error_layer1 = np.zeros(len(self.layer1.neurons))
        for j in range(len(self.layer1.neurons)):
            error_layer1[j] = 0
            for k in range(len(error_layer2)):
                error_layer1[j] += error_layer2[k] * self.layer2.neurons[k].weights[j]
            error_layer1[j] *= self.activation1.derivative(self.layer1.output[j])
        

        #update weights and biases
        #output layer
        for j in range(len(self.layer3.neurons)):
            self.layer3.neurons[j].weights = self.layer3.neurons[j].weights - eta * error_output[j]\
                                           * self.activation2.forward(self.layer2.output)
            self.layer3.neurons[j].bias = self.layer3.neurons[j].bias - eta * error_output[j]
        
        #second layer
        for j in range(len(self.layer2.neurons)):
            self.layer2.neurons[j].weights = self.layer2.neurons[j].weights - eta * error_layer2[j]\
                                           * self.activation1.forward(self.layer1.output)
            self.layer2.neurons[j].bias = self.layer2.neurons[j].bias - eta * error_layer2[j]

        #first layer, but in formula these weights and biases are not supposed to be edited, wtf?
        for j in range(len(self.layer1.neurons)):
            break 
            self.layer1.neurons[j].weights = self.layer1.neurons[j].weights - eta * error_layer1[j]\
                                           * data_input
            self.layer1.neurons[j].bias = self.layer1.neurons[j].bias - eta * error_layer1[j]
        



if __name__ == '__main__':
    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x)

    nn = NeuralNetwork(len(x), 4, 4)
    Nepochs = 1000
    for i in range(Nepochs):
        nn.forward(x)
        nn.backprop(y, x)
    nn.forward(x)

    import matplotlib.pyplot as plt
    plt.plot(x, y)
    plt.plot(x, nn.output)
    plt.show()
