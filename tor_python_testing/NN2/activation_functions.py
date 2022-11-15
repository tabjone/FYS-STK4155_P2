import numpy as np


class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

    def derivative(self, inputs):
        self.derivative_output = (inputs > 0) * 1


class ActivationSoftmax:
    def forward(self, inputs):
        #subtracting max value to stop overflow
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        #exp_values = np.exp(inputs) 

        return exp_values / np.sum(exp_values, axis=1, keepdims=True)
    
    def derivative(self, inputs):
        pass

class ActivationSigmoid:
    def forward(self, inputs):
        self.output = (inputs < 0) * np.exp(inputs)/(1+np.exp(inputs)) + (inputs >= 0) * 1/(1+np.exp(-inputs))
    
    def derivative(self, inputs):
        self.derivative_output = (1-self.output)*inputs
