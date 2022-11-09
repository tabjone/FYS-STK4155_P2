import numpy as np
from numpy.random import Generator, PCG64
rg = Generator(PCG64(1234))

import matplotlib.pyplot as plt

from activation_functions import *
from loss_functions import *


#SOFTMAX activation function
#to get rid of negative values we can do exponentiation and normalization
#y = exp(x), and y/np.sum(y)


#create practice dataset
def create_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes)

    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0,1, points)
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y


#without an activation function (only weights and biases) we would basically get a linear activation function
# is ReLU better than sigmoid?
#ReLU is aaaaalmost linear, but rectified. It's as powerful as sigmoid, but faster


#np.random.seed(0)

#Usual to call input data for X

#inputs = []
#weights = []
#bias = ...

#Alle neurons i et layer har samme inputs. Men forskjellige weights and biases:
#DENSE in NN means that the neuron is connected to all neurons of previous layer, and all of next layer


# z_l is the output of layer l
# a_l is the output of the activation function of layer l

#def backprop_error(error_next_layer, weights_next_layer):
    #for a single neuron
    # error = sum over k of (error for neuron k in next_layer * weight nr j of neuron k for next layer) multiplied by derivative_activation(output this layer)




def backprop_error_final_layer():
    #derivative_activation(output_layer_l) * gradient(activation_output_layer_l)
    pass


class Layer_Dense():
    def __init__(self, n_inputs, n_neurons):
        #there are two ways to init neural networks.
        #If we have a trained model: We load the weights and biases
        #If we don't, we initialize weights and biases
        # weights should be between -1 and 1. Smaller is better here because we want weights to tend towards small values. Or else it can explode
        #Therefore it is usual to scale dataset also so that input is between -1 and 1.
        #Usual to initialize biases as zero. But this can also create dead neurons which can kill the network. So be careful
        
        #number of inputs, number of neurons
        #making this (n_inputs, n_neurons) in shape makes us not need to transpose in the forward pass
        self.weights = rg.uniform(0, 2, (n_inputs, n_neurons)) - 1
        self.biases = np.zeros((1, n_neurons))
        
        #np.random.randn(n_inputs, n_neurons)

    def forward(self, inputs):
        #this is called z_l in the lecture notes, output for layer l
        self.output = np.dot(inputs, self.weights) + self.biases










#we choose cost-function sum(y-t)^2
def cost_OLS(predicted, target):
    return 1/2 * np.sum((prediced-target)**2)

def derivative_cost_OLS(predicted, target):
    return (targets-predicted)/(predicted*(1-predicted))


class neural_network:
    def __init__(self, X, h):
        self.X = X
        self.h = h
        #X is input data, h is number of neurons per layer
        self.layer1 = Layer_Dense(np.size(X,axis=1), h)
        self.activation1 = Activation_ReLU()

        self.layer2 = Layer_Dense(h, np.size(X,axis=1))
        self.activation2 = Activation_ReLU()
        
    def forward(self):
        self.layer1.forward(X)
        self.activation1.forward(self.layer1.output)
        self.layer2.forward(self.activation1.output)
        self.activation2.forward(self.layer2.output)

        self.output = self.activation2.output

    def backprop(self, target):
        eta = 0.001
        delta_L = self.activation2.derivative(self.layer2.output)* derivative_cost_OLS(self.activation2.output, target)
        
        delta_1 = 0
        for k in range(0, len(delta_L)):
            delta_1 += delta_L[k] * self.layer2.weights[k,:] * self.activation1.derivative(self.layer1.output)
       
        self.layer2.weights = self.layer2.weights - eta * delta_L * self.activation1




#COST IS CALCULATED AT LAST LAYER OF NETWORK. OUTPUT VS REAL DATA (TARGET)

def simple_cost(output, taget):
    return (output-target)**2


#y is target
#X is "data


X = np.zeros((100, 1))
X[:,0] = np.linspace(0,2*np.pi, 100)
y = np.sin(X[:,0])

NN = neural_network(X, 4)
NN.forward()
print(NN.output)

#print(len(X))
#print(np.size(X,axis=1))
"""

import nnfs
from nnfs.datasets import spiral_data

#X,y = spiral_data(samples=100, classes=3)
print(X.shape)
print(y.shape)

#fig, ax = plt.subplots()
#plt.scatter(X[:,0],y)
#plt.show()

#print(len(X[:,0]))
#print(X.shape)
#X,y = create_data(100, 3)


#data input
#X = [[1,2,3,4], 
#     [4,5,6,1],
#     [1,2,3,4]]

#100 neurons to handle 4 input data
dense1 = Layer_Dense(1, 90)
activation1 = Activation_ReLU()

#3 neurons to handle output from previous neuron
dense2 = Layer_Dense(90, 1)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)



dense2.forward(activation1.output)
activation2.forward(dense2.output)
#print(activation2.output)

#loss_function = Loss_CategoricalCrossentropy()
#loss = loss_function.calculate(activation2.output, y)

#n_inputs = 4
#n_neurons = 3
#print(np.random.randn(n_inputs, n_neurons).shape)

#print(rg.uniform(0, 2, (n_inputs, n_neurons))-1)
"""
