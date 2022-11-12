from gradient_methods import *

def learning_rate_decay(t, t0, t1):
    return t0/(t+t1)

if __name__ == '__main__':

    learning_rate_schedule = lambda t : learning_rate_decay(t, t0=3, t1=4)

    """
    # the number of datapoints
    n = 100
    #this is our input data x
    x = 2*np.random.rand(n,1)
    #this is the true data y
    y = 4+3*x+np.random.randn(n,1)
    #design matrix
    X = np.c_[np.ones((n,1)), x]

    #Ridge parameter lambda
    lmbda  = 0.001

i   #learning rate
    eta = 0.001
    #tolerance
    epsilon = 0.001
    #print("eigenvalues ")
    Niterations = 1000

    #plotting data
    plt.plot(x,y,'ro', label='data')

    #choosing random first value, this is normal
    first_value = np.random.randn(2,1)

    #beta values for plain GD
    beta_plain_ols = gradient_decent(X, y, eta, Niterations, epsilon, first_value, regression_method='OLS')	
    beta_plain_ridge = gradient_decent(X, y, eta, Niterations, epsilon, first_value, lmbda=lmbda, regression_method='Ridge')

    #predicted values plain GD
    ypredict_plain_ols = X @ beta_plain_ols
    ypredict_plain_ridge = X @ beta_plain_ridge

    #beta values for momentum GD
    beta_mgd_ols = gradient_decent(X, y, eta, Niterations, epsilon, first_value, regression_method='OLS', momentum_GD=True)	
    beta_mgd_ridge = gradient_decent(X, y, eta, Niterations, epsilon, first_value, lmbda=lmbda, regression_method='Ridge', momentum_GD=True)	

    #predicted values momentum GD
    ypredict_mgd_ols = X @ beta_mgd_ols
    ypredict_mgd_ridge = X @ beta_mgd_ridge

    #Beta values for stochastic GD with momentum
    Nepochs = 20

    first_beta_sgd_ols = first_value
    first_beta_sgd_ridge = first_value
    for epoch in range(Nepochs):
        first_beta_sgd_ols = gradient_decent(X, y, eta, Niterations, epsilon, first_beta_sgd_ols, \
                regression_method='OLS', momentum_GD=True, stochastic=True, batch_size=4)
        first_beta_sgd_ridge = gradient_decent(X, y, eta, Niterations, epsilon, first_beta_sgd_ridge, lmbda=lmbda,\
                regression_method='Ridge', momentum_GD=True, stochastic=True, batch_size=4)

    beta_sgd_ols = first_beta_sgd_ols
    beta_sgd_ridge = first_beta_sgd_ridge

    #predicted values stochastic momentum GD
    ypredict_sgd_ols = X @ beta_sgd_ols
    ypredict_sgd_ridge = X @ beta_sgd_ridge

    #plotting
    plt.plot(x,ypredict_plain_ols,label='plain ols')
    plt.plot(x,ypredict_plain_ridge,label='plain ridge')

    plt.plot(x, ypredict_mgd_ols, label='momentum GD ols')
    plt.plot(x, ypredict_mgd_ridge, label='momentum GD Ridge')

    plt.plot(x, ypredict_sgd_ols, label='stochastic OLS')
    plt.plot(x, ypredict_sgd_ridge, label='stochastic Ridge')

    plt.legend()
    plt.show()

    """
