#
# Every input, x is an ndarray type.

import numpy as np

def identity(x):
    return x

def step(x):
    return np.array(x > 0, dtype=np.int32)

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    return (1 - sigmoid(x)) * sigmoid(x)


def tanh(x):
    e1 = np.exp(x)
    e2 = np.exp(-x)
    return (e1 - e2) / (e1 + e2)

def softmax(x):          # x.shape: (records, features)
    if x.ndim==2:
        x_ = x.T         # to preserve records while broadcasting
        C = np.max(x_, axis=0)
        x_ = x_ - C      # to prevent overflow
        xp = np.exp(x_)  # xp.shape: (features, records)
        return (xp / np.sum(xp, axis=0)).T  # reverse the shape
    elif x.ndim ==1:
        C = np.max(x)
        xp = np.exp(x - C)
        return xp / np.sum(xp)

def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t)**2, axis=0)

def cross_entropy_error(y, t):
    if  y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def softmax_loss(x, t):
    y = softmax(x)
    return cross_entropy_error(y, t)