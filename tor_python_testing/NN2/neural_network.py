from layer import Layer, Neuron
from activation_functions import *

"""
Cost function: C= - 0.5*(target - predicted)^2
dC/dtarget = t-p
"""
class NeuralNetwork:
    def __init__(self, n_inputs, n_layers, n_neurons):
        self.n_layers = n_layers

        self.layers = []
        self.activations = []
        
        #first layer
        self.layers.append(Layer(n_inputs, n_neurons))
        self.activations.append(ActivationSigmoid())
        #hidden layers
        for i in range(1, n_layers-1):
            self.layers.append(Layer(len(self.layers[-1].neurons), n_neurons))
            self.activations.append(ActivationSigmoid())
        #output layer
        self.layers.append(Layer(len(self.layers[-1].neurons), n_inputs))
        self.activations.append(ActivationSigmoid())

    def forward(self, inputs):
        for l in range(len(self.layers)):
            self.layers[l].forward(inputs)
            self.activations[l].forward(self.layers[l].output)
            inputs = self.activations[l].output
            self.activations[l].derivative(self.layers[l].output)
        self.output = inputs

        
    def backprop(self, target, data_input):
        eta = 0.001
        
        #from last to first
        error_layers = []

        #error output
        error_output = np.zeros(len(self.layers[-1].neurons))
        for j in range(len(error_output)):
            error_output[j] = self.activations[-1].derivative_output[j] * (target[j] - self.output[j])
        
        error_layers.append(error_output)
        
        #error for hidden layers
        for l in range(len(self.layers)-2, 0, -1):
            error = np.zeros(len(self.layers[l].neurons))
            for j in range(len(error)):
                error[j] = 0
                for k in range(len(error_layers[-1])):        
                    error[j] += error_layers[-1][k] * self.layers[l+1].neurons[k].weights[j]
                error[j] *= self.activations[l].derivative_output[j]
            error_layers.append(error)
        
        error_input = np.zeros(len(self.layers[0].neurons))
        for j in range(len(self.layers[0].neurons)):
            error_input[j] = 0
            for k in range(len(error_layers[-1])):
                error_input[j] += error_layers[-1][k] * self.layers[1].neurons[k].weights[j]
            error_input[j] *= self.activations[0].derivative_output[j]
        error_layers.append(error_input)
        
        #flipping from first to last instead, for updating weights, biases
        error_layers.reverse()
        
        #updating weights and biases
        for l in range(len(self.layers)):
            for j in range(len(self.layers[l].neurons)):
                self.layers[l].neurons[j].weights = self.layers[l].neurons[j].weights - eta * error_layers[l][j]\
                                                  * self.activations[l].output[j]

                self.layers[l].neurons[j].bias = self.layers[l].neurons[j].bias - eta * error_layers[l][j]


def MSE(target, predicted):
    return np.sum((target-predicted)**2)


if __name__ == '__main__':
    x = np.linspace(0, 2*np.pi, 100)
    y = x
    
    n_layers_ = 4
    n_neutrons_ = 5
    nn = NeuralNetwork(len(x), n_layers=n_layers_, n_neurons=n_neutrons_)
    Nepochs = 15
    
    accuracy = []
    for i in range(Nepochs):
        nn.forward(x)
        nn.backprop(y, x)
        accuracy.append(MSE(nn.output, y))
    nn.forward(x)
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,2)
    fig.suptitle('{} epochs, {} layer, {} neurons/layer'.format(Nepochs, n_layers_, n_neutrons_))
    ax[0].plot(x, y, label='data')
    ax[0].plot(x, nn.output, label='nn output last epoch')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].legend()

    ax[1].plot(np.arange(0,Nepochs), accuracy)
    ax[1].set_xlabel('Nepochs')
    ax[1].set_ylabel('MSE')
    fig.tight_layout()
    plt.show()
