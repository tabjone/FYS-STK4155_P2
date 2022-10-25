from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sys

def gradient_ols(X, y, beta):
	n = len(y)

	return 2.0/n*X.T @ (X @ (beta)-y) 

def gradient_ridge(X, y, beta, lmbda):
	n = len(y)

	return 2.0/n*X.T @ (X @ (beta)-y)+2*lmbda*beta




"""Regression methods: OLS, Ridge"""

def beta_ols_regression(X, y):
	"""
	Calculates beta-parameters using OLS regression
	"""
	return np.linalg.inv(X.T @ X) @ X.T @ y

def beta_ridge_regression(X, y, lmbda):
	"""
	Calculates beta-parameters using Ridge regression
	"""
	XT_X = X.T @ X
	n = len(y)
	Id = n*lmbda* np.eye(XT_X.shape[0])
	return np.linalg.inv(XT_X + Id) @ X.T @ y

"""Plain Gradient decent methods: OLS, Ridge"""

def plain_gradient_decent_ols(X, y, eta, Niterations, epsilon):
	n = len(y)
	#guess first beta-value
	beta = np.random.randn(2,1)
	#creating inf first step-size to start while loop
	step_size = np.array([np.inf])	
	#Iterating untill step-size is smaller than epsilon or max number of iterations is reached
	iter = 0
	while (iter < Niterations) and (np.linalg.norm(step_size) >= epsilon): 
			gradients = gradient_ols(X, y, beta)
			step_size = eta*gradients
			beta -= step_size
			iter += 1
	return beta
	

def plain_gradient_decent_ridge(X, y, lmbda, eta, Niterations, epsilon):
	#guess first beta-value
	beta = np.random.randn(2,1)
	#creating inf first step-size to start while loop
	step_size = np.array([np.inf])	
	#Iterating untill step-size is smaller than epsilon or max number of iterations is reached
	iter = 0
	while (iter <  Niterations) and (np.linalg.norm(step_size) >= epsilon): 
		gradients = gradient_ridge(X, y, beta, lmbda)
		step_size = eta*gradients
		beta -= step_size
		iter += 1
	return beta
	
"""Momentum gradient decent"""
def momentum_gradient_decent_ols(X, y, eta, Niterations, epsilon, momentum):
	#guess first solution
	solution = np.random.randn(2,1)
	#creating inf first step-size to start while loop
	step_size = np.array([np.inf])	
	previous_step_size = 0

	#run the gradient recent
	iter = 0
	while (iter <  Niterations) and (np.linalg.norm(step_size) >= epsilon): 
		#calculate the gradient
		gradient = gradient_ols(X, y, solution)
		step_size = eta*gradient

		#take a step
		solution += - step_size + momentum * previous_step_size

		iter += 1
		previous_step_size = step_size

	return solution

def momentum_gradient_decent_ridge(X, y, lmbda,  eta, Niterations, epsilon, momentum):
	#guess first solution
	solution = np.random.randn(2,1)
	#creating inf first step-size to start while loop
	step_size = np.array([np.inf])	
	previous_step_size = 0

	#run the gradient recent
	iter = 0
	while (iter <  Niterations) and (np.linalg.norm(step_size) >= epsilon): 
		#calculate the gradient
		gradient = gradient_ridge(X, y, solution, lmbda) 

		step_size = eta*gradient

		#take a step
		solution += - step_size + momentum * previous_step_size

		iter += 1
		previous_step_size = step_size

	return solution

#epoch is outside this. This is a "full run" of finding beta or solution. Then we run again with ?solution as intitial guess? and do this again and again for as many epochs as we want.
"""Stochastic gradient decent, plain"""
def sgd_ols(X, y, eta, Niterations, epsilon, batch_size):
	"""
	let's say X has m rows and n columns. Now we split those m columns of X into batch_size number of batches. Then we pick a random batch to do the gradient.
	"""
	#splitting array
	X = np.array_split(X, batch_size, axis=0) #this will split into as even parts as it can
	y = np.array_split(y, batch_size)
	
	solution = np.random.randn(2,1)
	step_size = np.array([np.inf])

	iter = 0
	while (iter < Niterations) and (np.linalg.norm(step_size) >= epsilon):
		batch_nr = np.random.randint(batch_size)
		mini_batch = X[batch_nr]
		gradient = gradient_ols(mini_batch, y[batch_nr], solution) 
		step_size = eta*gradient
		solution += - step_size

		iter += 1
	return solution



"""Add momentum to the plain GD code and compare convergence with a fixed learning rate (you may need to tune the learning rate)."""


"""
Notes:
It is usual to start with a large learning rate and make it smaller with each step. The way the learning rate changes from big to small is called the scedule
Momentum=0.9 is regular

mini-batches should be in powers of 2. Makes it computationally faster or something

"""

if __name__ == '__main__':
	# the number of datapoints
	n = 100
	x = 2*np.random.rand(n,1)
	y = 4+3*x+np.random.randn(n,1)

	X = np.c_[np.ones((n,1)), x]

	#Ridge parameter lambda
	lmbda  = 0.001


	XT_X = X.T @ X
	# Hessian matrix
	H = (2.0/n)* XT_X+2*lmbda* np.eye(XT_X.shape[0])
	# Get the eigenvalues
	EigValues, EigVectors = np.linalg.eig(H)
	#print(f"Eigenvalues of Hessian Matrix:{EigValues}")

	eta = 1.0/np.max(EigValues)
	
	epsilon = 0.001
	#print("eigenvalues ")
	Niterations = 1000
	beta_sgd_ols = sgd_ols(X, y, eta, Niterations, epsilon, batch_size=3)

	plt.plot(x,y,'ro')
	plt.plot(x, X @ beta_sgd_ols)
	plt.show()
	"""
	beta_plain_ols = plain_gradient_decent_ols(X, y, eta, Niterations, epsilon)	
	beta_plain_ridge = plain_gradient_decent_ridge(X, y, lmbda, eta, Niterations, epsilon)
	
	ypredict_plain_ols = X @ beta_plain_ols
	ypredict_plain_ridge = X @ beta_plain_ridge
	

	beta_mgd_ols = momentum_gradient_decent_ols(X, y, eta, Niterations, epsilon, momentum=0.9) 
	beta_mgd_ridge = momentum_gradient_decent_ridge(X, y, lmbda,  eta, Niterations, epsilon, momentum=0.9)

	ypredict_mgd_ols = X @ beta_mgd_ols
	ypredict_mgd_ridge = X @ beta_mgd_ridge

	
	plt.plot(x, ypredict_plain_ols, label='plain OLS')
	plt.plot(x, ypredict_plain_ridge, label='plain Ridge')

	plt.plot(x, ypredict_mgd_ols, label='MGD OLS')
	plt.plot(x, ypredict_mgd_ridge, label='MGD Ridge')

	plt.plot(x, y, 'ro', label='Solution')
	plt.legend()
	plt.show()
	"""
	"""	
	beta_linreg = beta_ridge_regression(X, y, lmbda)
	ypredict = X @ beta
	ypredict2 = X @ beta_linreg
	plt.plot(x, ypredict, "r-")
	plt.plot(x, ypredict2, "b-")
	plt.plot(x, y ,'ro')
	plt.axis([0,2.0,0, 15.0])
	plt.xlabel(r'$x$')
	plt.ylabel(r'$y$')
	plt.title(r'Gradient descent example for Ridge')
	plt.show()
	"""
