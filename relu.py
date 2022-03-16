import numpy as np

def relu(z):
    return np.where(z > 0, z, 0)

def relu_prime(z):
    return np.where(z > 0, 1, 0)