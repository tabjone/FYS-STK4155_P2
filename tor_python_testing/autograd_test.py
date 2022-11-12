import autograd.numpy as np
from autograd import grad

if __name__ == '__main__':
    epsilon = 1e-8
    eta = 1e-3
    beta = 0.9

    first_moment = gradients(...)
    second_moment = beta*last_second_moment + (1-beta)*(first_moment)**2
    
    theta += eta * first_moment/np.sqrt(second_moment + epsilon)


    """
    Multiplication and division by vectors is understood as an element-wise operation.
    It is clear from this formula that the learning rate is reduced in directions where the norm of the gradient is consistently large.
    This greatly speeds up the convergence by allowing us to use a larger learning rate for flat directions.
    """
