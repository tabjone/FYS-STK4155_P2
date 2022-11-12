from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sys

class GradientDecent:
    def __init__(self, alpha=0, lmbda=0):
        """Input alpha is momentum parameter
           lmbda is Ridge hyperparameter"""
        #setting memory term=0 for first iteration
        self.v = 0
        self.alpha=alpha
        self.lmbda = lmbda
        #setting this =infty for first iteration for stopping criterion of while loop
        self.g = np.infty

    def compute_gradient(self, X, y, beta):
        """Loss/Cost-function is OLS if lmbda=0 and Ridge if lmbda!=0"""
        m = len(y)
        return 2.0/m*X.T @ (X @ (beta)-y)+2*self.lmbda*beta
    
    def iterate(self, X, y, eta):
        """eta is learning rate"""
        #calculate gradient
        self.g = self.compute_gradient(X, y, self.theta)
        #compute velocity update
        self.v = self.alpha * self.v - eta * self.g
        #apply update
        self.theta += self.v
        
    def solve(self, X, y, initial_solution, Niterations, eta=0.01, epsilon=0.001):
        """Niterations is max number of iters, eta is learning rate, epsilon is stopping criterion"""
        self.theta = initial_solution
        self.X = X
        self.y = y

        iter = 0
        while iter <= Niterations and abs(np.linalg.norm(eta*self.g)) >= epsilon:
            self.iterate(X, y, eta)
            iter += 1
            
    def get_solution(self):
        """gets predicted solution """
        return self.X @ self.theta


class StochasticGradientDecent(GradientDecent):
    def solve(self, X, y, initial_solution, Nepochs, size_minibatch, learning_schedule=None, eta=None, epsilon=0.001):
        self.theta = initial_solution
        self.X = X
        self.y = y

        M = size_minibatch   #size of each minibatch
        m = int(n/M) #number of minibatches
        
        #returning eta if no learning schedule is set.
        if learning_schedule==None: 
            learning_schedule = lambda t : eta
            if eta == None:
                raise TypeError('Must include learning schedule or learning rate')

        epoch = 0
        eta = learning_schedule(0)
        while epoch <= Nepochs and abs(np.linalg.norm(eta*self.g)) >= epsilon:
            for i in range(m):
                random_index = M*np.random.randint(m)
                xi = X[random_index:random_index+M]
                yi = y[random_index:random_index+M]
                eta = learning_schedule(epoch*m+i)
                self.iterate(xi, yi, eta)
            epoch += 1

#THIS CAN BE COMBINED IN SOME WAY WITH REGULAR SGD, ADD TWO ITERATE METHODS TO REGULAR GD CLASS, ITERATE_ADAGRAD AND ITERATE
class SGD_AdaGrad(GradientDecent):
    def iterate(self, X, y, eta):
        delta = 1e-7
        """eta is learning rate"""
        #calculate gradient
        self.g = self.compute_gradient(X, y, self.theta)
        #Accumulate squared gradient
        self.r += self.g * self.g 
        #Compute parameter update
        dtheta = - eta /(delta + np.sqrt(self.r)) * self.g
        #apply update
        self.theta += dtheta

    def solve(self, X, y, initial_solution, Nepochs, size_minibatch, eta=0.01, epsilon=0.001):
        self.theta = initial_solution
        self.X = X
        self.y = y

        M = size_minibatch   #size of each minibatch
        m = int(n/M) #number of minibatches
        
        self.r = 0 #Initialize accumulation variables
        epoch = 0
        eta = learning_schedule(0)
        while epoch <= Nepochs and abs(np.linalg.norm(eta*self.g)) >= epsilon:
            for i in range(m):
                random_index = M*np.random.randint(m)
                xi = X[random_index:random_index+M]
                yi = y[random_index:random_index+M]
                self.iterate(xi, yi, eta)
            epoch += 1


def learning_schedule_decay(t, t0, t1):
    return t0/(t+t1)

def MSE(predicted, target):
    return np.sum((predicted-target)**2)


"""
Task: 
-Use a tunable learning rate as discussed in the lectures from week 39."
-Implement the Adagrad method in order to tune the learning rate. Do this with and without momentum for plain gradient descent and SGD.
-Add RMSprop and Adam to your library of methods for tuning the learning rate.
-Add momentum to the plain GD code and compare convergence with a fixed learning rate (you may need to tune the learning rate).
"""


import copy


if __name__ == '__main__':

    # the number of datapoints
    n = 100

    # y as second order polynomial
    x = np.linspace(0, 5, n).reshape(-1,1)
    a0 = 1; a1 = 4; a2 = 1
    y = a0 + a1*x + a2*x**2
    X = np.c_[np.ones((n,1)), x, x**2]
    #X_ = copy.deepcopy(X)
    #y_ = copy.deepcopy(y)

    #y as random numbers
    #x = 2*np.random.rand(n,1)
    #y = 4+3*x+np.random.randn(n,1)

    #X = np.c_[np.ones((n,1)), x]
    
    #Ridge parameter lambda
    #lmbda  = 0.001
    
    initial_solution = (np.random.randn(X.shape[1],1))
    
    #### HER SETTER JEG OPP ETT OBJEKT TIL KLASSEN GD OG LØSER
    gd_plain = GradientDecent()
    gd_plain.solve(X, y, initial_solution=initial_solution, Niterations=1000, eta=0.001)
    
    A = gd_plain.get_solution()

    #print(gd_plain.get_solution())

    #### HER SETTER JEG OPP ETT OBJEKT TIL KLASSEN SGD OG LØSER
    learning_schedule = lambda t : learning_schedule_decay(t, t0=5, t1=50) 
    sgd= StochasticGradientDecent()
    sgd.solve(X, y, initial_solution=initial_solution, Nepochs=200, size_minibatch=20, learning_schedule=learning_schedule)
    
    B = gd_plain.get_solution()
    print(A == B)
    

    #plt.plot(x, y, label='data')
    #plt.plot(x, gd_plain.get_solution, label='gd plain')
    #plt.show()

