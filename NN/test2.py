import numpy as np
from neuron_2 import *
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# one-hot in numpy
def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1
    
    return onehot_vector



if __name__ == '__main__':
    
    size = 1000
    X = np.zeros((size,1))
    X[:,0] = np.linspace(0,2*np.pi, size)
    y = np.sin(X)
    #print(X)
    #nn = NeuralNetwork(X, y, n_categories=1)
    #nn.train()

    #X_pred = nn.predict(X)
    #print(X_pred)
    #plt.plot(X[:,0], nn.predict(X[:,0]))

    train_size = 0.8
    test_size = 1 - train_size
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=train_size,
                                                        test_size=test_size)

    #Y_train_onehot, Y_test_onehot = to_categorical_numpy(Y_train), to_categorical_numpy(Y_test)
    epochs = 10
    batch_size = 2
    eta=0.001
    lmbd = 0.01

    #dnn = NeuralNetwork(X_train, Y_train_onehot, epochs=epochs, batch_size=batch_size, eta=eta, lmbd=lmbd)
    #dnn = NeuralNetwork(X_train, Y_train, epochs=epochs, batch_size=batch_size, eta=eta, lmbd=lmbd, n_categories=2, n_hidden_neurons=50)
    dnn = NeuralNetwork(X_train, Y_train)
    dnn.train()
    test_predict = dnn.predict_probabilities(X)
    #print(y.shape)
    #print(test_predict.shape)
    #print(test_predict)
    plt.plot(X[:,0], y[:,0], label='real')
    plt.plot(X[:,0], test_predict[:,0], label='nn')
    plt.legend()
    plt.show()
    

"""
# download MNIST dataset
digits = datasets.load_digits()

# define inputs and labels
inputs = digits.images
labels = digits.target

print("inputs = (n_inputs, pixel_width, pixel_height) = " + str(inputs.shape))
print("labels = (n_inputs) = " + str(labels.shape))


# flatten the image
# the value -1 means dimension is inferred from the remaining dimensions: 8x8 = 64
n_inputs = len(inputs)
inputs = inputs.reshape(n_inputs, -1)
print("X = (n_inputs, n_features) = " + str(inputs.shape))


# choose some random images to display
indices = np.arange(n_inputs)
random_indices = np.random.choice(indices, size=5)

for i, image in enumerate(digits.images[random_indices]):
    plt.subplot(1, 5, i+1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title("Label: %d" % digits.target[random_indices[i]])
plt.show()

# one-liner from scikit-learn library
train_size = 0.8
test_size = 1 - train_size
X_train, X_test, Y_train, Y_test = train_test_split(inputs, labels, train_size=train_size,
                                                    test_size=test_size)



#Y_train_onehot, Y_test_onehot = to_categorical(Y_train), to_categorical(Y_test)
Y_train_onehot, Y_test_onehot = to_categorical_numpy(Y_train), to_categorical_numpy(Y_test)





epochs = 100
batch_size = 100
eta=0.001
lmbd = 0.01

dnn = NeuralNetwork(X_train, Y_train_onehot, epochs=epochs, batch_size=batch_size, eta=eta, lmbd=lmbd)
dnn.train()
test_predict = dnn.predict(X_test)

# accuracy score from scikit library
print("Accuracy score on test set: ", accuracy_score(Y_test, test_predict))
"""
