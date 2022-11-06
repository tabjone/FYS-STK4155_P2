import numpy as np
from activation_functions import *

class NeuralNetwork:
    def __init__(
            self,
            X_data,
            Y_data,
            n_hidden_neurons=50,
            n_categories=10,
            epochs=10,
            batch_size=100,
            eta=0.1,
            lmbd=0.0):

        self.X_data_full = X_data
        self.Y_data_full = Y_data

        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_hidden_neurons = n_hidden_neurons
        self.n_categories = n_categories

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd

        self.create_biases_and_weights()
        
        #setting up activation functions, these can be class inputs in future
        self.hidden_activation_func = Activation_Sigmoid()
        self.output_activation_func = Activation_Softmax()

    def create_biases_and_weights(self):
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons)
        self.hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01

        self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_categories)
        self.output_bias = np.zeros(self.n_categories) + 0.01

    def feed_forward(self):
        """feed forward for training"""
        #output of hidden layer
        self.z_h = np.matmul(self.X_data, self.hidden_weights) + self.hidden_bias
        #activation of hidden layer
        self.a_h = self.hidden_activation_func.forward(self.z_h)
        #output of last layer before activation
        self.z_o = np.matmul(self.a_h, self.output_weights) + self.output_bias
        #activation of last layer
        self.probabilities = self.output_activation_func.forward(self.z_o)

    def feed_forward_out(self, X):
        """feed-forward for output"""
        #output of hidden layers
        z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias
        #activation of hidden layers
        a_h = self.hidden_activation_func.forward(z_h)
        #output of last layer
        z_o = np.matmul(a_h, self.output_weights) + self.output_bias
        #activation of last layer
        probabilities = self.output_activation_func.forward(z_o)
        return probabilities

    def backpropagation(self):
        """See bottom of week40"""
        #this is delta^L (error of last layer)
        error_output = self.probabilities - self.Y_data
        #this is delta^l (error of hidden layer)
        error_hidden = np.matmul(error_output, self.output_weights.T) * self.hidden_activation_func.derivative(self.z_h)
        
        #updating weights and biases like described in bottom of week40
        self.output_weights_gradient = np.matmul(self.a_h.T, error_output)
        self.output_bias_gradient = np.sum(error_output, axis=0)

        self.hidden_weights_gradient = np.matmul(self.X_data.T, error_hidden)
        self.hidden_bias_gradient = np.sum(error_hidden, axis=0)
        
        #this is the additional update if we have Ridge. Idk why its like this.
        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.output_weights
            self.hidden_weights_gradient += self.lmbd * self.hidden_weights
        #gradient decent
        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias -= self.eta * self.output_bias_gradient
        self.hidden_weights -= self.eta * self.hidden_weights_gradient
        self.hidden_bias -= self.eta * self.hidden_bias_gradient

    def predict(self, X):
        probabilities = self.feed_forward_out(X)
        return np.argmax(probabilities, axis=1)

    def predict_probabilities(self, X):
        """This returns the predicted output of the neural network"""
        probabilities = self.feed_forward_out(X)
        return probabilities

    def train(self):
        data_indices = np.arange(self.n_inputs)
        #stochastic gd
        for i in range(self.epochs):
            for j in range(self.iterations):
                # pick datapoints with replacement
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )

                # minibatch training data
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]

                self.feed_forward()
                self.backpropagation()




if __name__ == '__main__':
    epochs = 100
    batch_size = 100

    dnn = NeuralNetwork(X_train, Y_train_onehot, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
                        n_hidden_neurons=n_hidden_neurons, n_categories=n_categories)
    dnn.train()
    test_predict = dnn.predict(X_test)

    # accuracy score from scikit library
    print("Accuracy score on test set: ", accuracy_score(Y_test, test_predict))

    # equivalent in numpy
    def accuracy_score_numpy(Y_test, Y_pred):
        return np.sum(Y_test == Y_pred) / len(Y_test)

    #print("Accuracy score on test set: ", accuracy_score_numpy(Y_test, test_predict))
