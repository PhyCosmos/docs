# coding: utf-8
# Every input, x is an ndarray type.
import sys, os
sys.path.append(os.pardir)
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
    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def softmax_loss(x, t):
    y = softmax(x)
    return cross_entropy_error(y, t)



# Gradients
def num_gradient_without_batch(f, x):
    """Gradient of a scalar valued function in a numerical way.
    x.size iteration required!
    At position x known in numerical values, for each i, we calculate
    f(xi + h) and f(xi - h), Hence, at least, 2 * x.size computations
    are needed.

    Parameters
    ----------
    f : scalar valued function of an array
    x : input array of any shape.

    Return
    ------
    gradient: An array of the shape of x
    """
    h = 1e-4     # small displacement
    grad = np.zeros_like(x)
    # iterate (the size of x) times
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        original = x[idx]
        # xi -> xi + h : forward
        x[idx] = float(original) + h
        fxh1 = f(x) # f(xi + 1)
        # xi -> xi - h : backward
        x[idx] = float(original) - h
        fxh2 = f(x) # f(xi - 1)
        # average difference: central difference
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = original # retrieve the original value

        it.iternext()

    return grad

def num_gradient(f, X):
    """Gradient of a scalar valued function in a numerical way.
    Since X may be a batch data or not, reshaping X and iteration by batch_size
    are required.

    Parameters
    ----------
    f : scalar valued function of an array
    X : input array of any shape with "first dimension of batch_size".

    Return
    ------
    gradient: An array of the shape of X
    """

    if X.ndim == 1:
        # X = X.reshape(1, X.size)
        return num_gradient_without_batch(f, X)    
    grad = np.zeros_like(X)

    for i, x in enumerate(X):    # for each record x in a batch X
        grad[i] = num_gradient_without_batch(f, x)

    return grad