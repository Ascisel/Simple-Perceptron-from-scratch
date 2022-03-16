import numpy as np
from network import Network
from hiddenlayer import HiddenLayer
from outputlayer import OutputLayer
from mse import mse_prime
from relu import relu, relu_prime
from tanh import tanh, tanh_prime
from labels_converter import convert_labels

if __name__ == "__main__":
    #Load data from MNIST dataset

    train_data = np.loadtxt("mnist_train.csv", 
                            delimiter=",")
    test_data = np.loadtxt("mnist_test.csv", 
                            delimiter=",") 


    # Convert loaded data to float arrays
    # Multiply each element in array by 0.99 / 255 so that
    # each element is in interval [0, 0.99] and add 0.01 so that
    # none of elements is equal to 0

    factor = 1 / 255
    train_imgs = np.asfarray(train_data[:, 1:]) * factor
    test_imgs = np.asfarray(test_data[:, 1:]) * factor
    train_imgs = train_imgs.reshape(train_imgs.shape[0], 1, 28*28)
    test_imgs = test_imgs.reshape(test_imgs.shape[0], 1, 28*28)

    train_labels = np.asfarray(train_data[:, :1])
    test_labels = np.asfarray(test_data[:, :1])

    #train_imgs = train_imgs.reshape(train_imgs.shape[0], 1, 28*28)
    train_labels = convert_labels(train_labels)
    test_labels = convert_labels(test_labels)

    hidden = HiddenLayer(28*28, 100, tanh, tanh_prime)
    output = OutputLayer(100, 10)
    ournetwork = Network(hidden, output)
    ournetwork.train(train_imgs[:1000], train_labels[:1000], 1, 0.1, mse_prime)

    out = ournetwork.predict(test_imgs[0:3])
    print("\n")
    print("predicted values : ")
    print(out, end="\n")
    print("true values : ")
    print(test_labels[0:3])

