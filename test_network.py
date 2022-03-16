import numpy as np
from NeuronNetwork import NeuronNetwork
from hiddenlayer import HiddenLayer
from outputlayer import OutputLayer
from mse import mse_prime
from exp import exp_act, exp_act_prime
from tanh import tanh, tanh_prime
from work_with_data import convert_labels


def prepare_data(): # przygotowanie danych do testowania perceptronu
    train_data = np.loadtxt("mnist_train.csv", 
                            delimiter=",")
    test_data = np.loadtxt("mnist_test.csv", 
                            delimiter=",") 

    factor = 0.99 / 255
    train_imgs = np.asfarray(train_data[:, 1:]) * factor + 0.01
    test_imgs = np.asfarray(test_data[:, 1:]) * factor + 0.01
    train_imgs = train_imgs.reshape(train_imgs.shape[0], 1, 28*28)
    test_imgs = test_imgs.reshape(test_imgs.shape[0], 1, 28*28)

    train_labels = np.asfarray(train_data[:, :1])
    test_labels = np.asfarray(test_data[:, :1])

    train_labels = convert_labels(train_labels)
    test_labels = convert_labels(test_labels)

    return (train_imgs, train_labels), (test_imgs, test_labels)


def build_network(input_size, output_size, h_neutr, activ_func, activ_func_prime): # inicjacja sieci o zadanych parametrach
    hidden = HiddenLayer(input_size, h_neutr, activ_func, activ_func_prime)
    output = OutputLayer(h_neutr, output_size)
    ournetwork = NeuronNetwork(hidden, output)

    return ournetwork

if __name__ == "__main__":
    (train_x, train_y), (test_x, test_y) = prepare_data()
    network = build_network(28*28, 10, 100, exp_act, exp_act_prime)

    network.train(train_x, train_y, 0.1, mse_prime)

