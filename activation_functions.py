import numpy as np

# All of these need to be able to handle 2D numpy arrays as inputs.


def none(v):
    return v


def tanh(v):
    return np.tanh(v)


def logit(v):
    return 1 / (1 + np.exp(-v))


def relu(v):
    return np.maximum(0, v)
