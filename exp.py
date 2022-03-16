import numpy as np


def exp_act(x):
    return np.exp(x)/(1 + np.exp(x))


def exp_act_prime(x):
    return np.exp(x)/((1 + np.exp(x))**2)
