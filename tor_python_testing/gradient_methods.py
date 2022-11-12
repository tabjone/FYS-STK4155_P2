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

        iter = 0
        while iter <= Niterations and abs(np.linalg.norm(eta*self.g)) >= epsilon:
            self.iterate(X, y, eta)
            iter += 1
            
    def get_solution(self, X):
        """gets predicted solution """
        return X @ self.theta


class StochasticGradientDecent(GradientDecent):
    def solve(self, X, y, initial_solution, Nepochs, size_minibatch, learning_schedule=None, eta=None, epsilon=0.001):
        self.theta = initial_solution

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

if __name__ == '__main__':

    # the number of datapoints
    n = 100
    x = 2*np.random.rand(n,1)
    y = 4+3*x+np.random.randn(n,1)

    X = np.c_[np.ones((n,1)), x]
    
    #Ridge parameter lambda
    #lmbda  = 0.001

    gd_method = GradientDecent()
    gd_method.solve(X, y, initial_solution=np.random.randn(2,1), Niterations=1000, epsilon=0.001)
    
    y_gd_plain = gd_method.get_solution(X)

    plt.scatter(x,y, color='red',label='data')
    plt.plot(x, y_gd_plain, label='plain GD, no mentum, OLS')

    learning_schedule = lambda t : learning_schedule_decay(t, t0=5, t1=50) 

    gd_stochastic = StochasticGradientDecent()
    gd_stochastic.solve(X, y, initial_solution=np.random.randn(2,1), Nepochs=200, size_minibatch=20, learning_schedule=learning_schedule)
    
    y_sgd = gd_stochastic.get_solution(X)

    plt.plot(x, y_sgd, label='sgd, no momentum, OLS')
     
    plt.legend()
    plt.show()
    
    print('MSE GD: {}'.format(MSE(y_gd_plain, y)))
    print('MSD SGD: {}'.format(MSE(y_sgd, y)))
