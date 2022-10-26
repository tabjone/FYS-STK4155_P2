from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sys

"""Closed-from Regression methods: OLS, Ridge"""
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


"""Gradients: OLS and Ridge"""
def gradient_ols(X, y, beta, lmbda=None):
	n = len(y)

	return 2.0/n*X.T @ (X @ (beta)-y) 

def gradient_ridge(X, y, beta, lmbda):
	n = len(y)

	return 2.0/n*X.T @ (X @ (beta)-y)+2*lmbda*beta


"""step-function for plain and momentum GD """
def step_plain_GD(step_size, momentum=None, previous_step_size=None):
	return - step_size

def step_momentum_GD(step_size, momentum, previous_step_size):
	return - step_size + momentum * previous_step_size


"""Calculate step size for stochastic or regular GD"""
def stochastic_step_size():
	#chosing random mini-batch and calculating step-size
	batch_nr = np.random.randint(batch_size)
	mini_batch = X[batch_nr]
	step_size = eta * gradient(mini_batch, y[batch_nr], solution, lmbda) 
	return step_size

def regular_step_size(X, y, gradient, solution, lmbda, eta, batch_size=None):
	step_size = eta*gradient(X, y, solution, lmbda)
	return step_size



"""GD OLS and Ridge"""
def gradient_decent(X, y, eta, Niterations, epsilon, initial_solution, momentum=0.9, lmbda=None, regression_method=None, momentum_GD=False, stochastic=False, batch_size=None):
	#setting up initial solution
	solution = initial_solution

	#getting gradient
	if regression_method.upper() == 'OLS': gradient = gradient_ols
	elif regression_method.upper() == 'RIDGE': gradient = gradient_ridge
	else: raise ValueError('Wrong input in regression_method')

	#getting step-function
	if momentum_GD: step = step_momentum_GD
	else: step = step_plain_GD
	
	#getting iteration step-size for stochastic or regular GD
	if stochastic:
		step_size = stochastic_step_size
		#splitting array
		X = np.array_split(X, batch_size, axis=0) 
		y = np.array_split(y, batch_size)
	else: step_size = regular_step_size

	#creating memory term
	previous_step_size = 0
	#creating inf first step-size to start while loop
	step_size = np.array([np.inf])	
	#Iterating untill step-size is smaller than epsilon or max number of iterations is reached
	iter = 0
	while (iter < Niterations) and (np.linalg.norm(step_size) >= epsilon): 
			step_size = eta*gradient(X, y, solution, lmbda)
			#take a step
			solution += step(step_size, momentum, previous_step_size)
			iter += 1
			#save step
			previous_step_size = step_size
	return solution





#epoch is outside this. This is a "full run" of finding beta or solution. Then we run again with ?solution as intitial guess? and do this again and again for as many epochs as we want.
"""Stochastic gradient decent, plain"""
def stochastic_gradient_decent(X, y, eta, Niterations, epsilon, initial_solution, momentum=0.9, lmbda=None, regression_method=None, GD_method=None, batch_size=None):
	#setting up initial solution
	solution = initial_solution

	#getting gradient
	if regression_method.upper() == 'OLS': gradient = gradient_ols
	elif regression_method.upper() == 'RIDGE': gradient = gradient_ridge
	else: raise ValueError('Wrong input in regression_method')

	#getting step-function
	if GD_method.upper() == 'PLAIN': step = step_plain_GD
	elif GD_method.upper() == 'MOMENTUM': step = step_momentum_GD
	else: raise ValueError('Wrong input in GD_method')

	#splitting array
	X = np.array_split(X, batch_size, axis=0) 
	y = np.array_split(y, batch_size)

	#creating memory term
	previous_step_size = 0
	#creating inf first step-size to start while loop
	step_size = np.array([np.inf])	
	#Iterating untill step-size is smaller than epsilon or max number of iterations is reached
	iter = 0
	while (iter < Niterations) and (np.linalg.norm(step_size) >= epsilon):
		#chosing random mini-batch and calculating step-size
		batch_nr = np.random.randint(batch_size)
		mini_batch = X[batch_nr]
		step_size = eta * gradient(mini_batch, y[batch_nr], solution, lmbda) 
		#take a step
		solution += step(step_size, momentum, solution)
		iter += 1
		#save step
		previous_step_size = step_size 
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


	plt.plot(x,y,'ro', label='data')

	first_value = np.random.randn(2,1)

	beta_plain_ols = gradient_decent(X, y, eta, Niterations, epsilon, first_value, regression_method='OLS', GD_method='plain')	
	beta_plain_ridge = gradient_decent(X, y, eta, Niterations, epsilon, first_value, lmbda=lmbda, regression_method='Ridge', GD_method='plain')
	
	ypredict_plain_ols = X @ beta_plain_ols
	ypredict_plain_ridge = X @ beta_plain_ridge


	beta_mgd_ols = gradient_decent(X, y, eta, Niterations, epsilon, first_value, regression_method='OLS', GD_method='momentum')	
	beta_mgd_ridge = gradient_decent(X, y, eta, Niterations, epsilon, first_value, lmbda=lmbda, regression_method='Ridge', GD_method='momentum')	

	ypredict_mgd_ols = X @ beta_mgd_ols
	ypredict_mgd_ridge = X @ beta_mgd_ridge


	beta_sgd_ols = stochastic_gradient_decent(X, y, eta, Niterations, epsilon, first_value, batch_size=3, regression_method='OLS', GD_method='momentum')
	beta_sgd_ridge = stochastic_gradient_decent(X, y, eta, Niterations, epsilon, first_value, batch_size=3, lmbda=lmbda, regression_method='Ridge', GD_method='momentum')

	plt.plot(x,ypredict_mgd_ols,label='1')
	plt.plot(x,ypredict_mgd_ridge,label='2')
	plt.legend()
	plt.show()

	"""
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
