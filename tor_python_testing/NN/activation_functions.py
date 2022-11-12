import numpy as np


class Activation_ReLU:
    def forward(self, inputs):
        return np.maximum(0, inputs)

    def derivative(self, inputs):
        return (inputs > 0) * 1


class Activation_Softmax:
    def forward(self, inputs):
        #subtracting max value to stop overflow
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        #exp_values = np.exp(inputs) 

        return exp_values / np.sum(exp_values, axis=1, keepdims=True)
    
    def derivative(self, inputs):
        pass

class Activation_Sigmoid:
    def forward(self, inputs):
        #return (inputs < 0) * np.exp(inputs)/(1+np.exp(inputs)) + (inputs >= 0) * 1/(1+np.exp(-inputs))
        return 1/(1+np.exp(-inputs))
    
    def derivative(self, inputs):
        return (1-self.forward(inputs))*inputs


