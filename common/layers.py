# coding: utf-8
import numpy as np
from common.functions import *


class Multiplication:   # swap multiplier
    def __init__(self) -> None:
        self.x = None 
        self.y = None
    def forward(self, x, y):
        self.x = x
        self.y = y
        return x * y
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy

class Maximum:          # graident router
    def __init__(self) -> None:
        self.maskx = None
        self.masky = None
    def forward(self, x, y):
        self.maskx = np.array(x >= y, dtype=np.int16)
        self.masky = np.array(x <= y, dtype=np.int16)
        return np.maximum(x, y)
    def backward(self, dout):
        dx = dout * self.maskx
        dy = dout * self.masky
        return dx, dy

class  Addition:     # gradient distributor
    def __init__(self) -> None:
        self.x = None
        self.y = None
    def forward(self, x, y):
        self.x = x
        self.y = y
        return x + y
    def backward(self, dout):
        dx = dout
        dy = dout
        return dx, dy

class Copy:         # gradient adder
    def __init__(self) -> None:
        self.x = None
        self.dout1 = None
        self.dout2 = None
    def forward(self, x):
        self.x = x
        self.dout1 = x.copy()
        self.dout2 = x.copy()
        return self.dout1, self.dout2
    def backward(self, dout1, dout2):
        return dout1 + dout2

class Step:
    def __init__(self) -> None:
        self.x = None
    def forward(self, x):
        self.x = x
        return np.array(x>0, dtype=np.int32)
    def backward(self, dout):
        return self.x * 0

class Relu:
    def __init__(self) -> None:
        self.mask = None
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    def backward(self, dout):
        dout[self.mask] = 0
        return dout

class Sigmoid:
    def __init__(self) -> None:
        self.out = None
    def forward(self, x):
        self.out = sigmoid(x)
        return self.out
    def backward(self, dout):
        return dout * (1.0 - self.out) * self.out

class Affine:
    def __init__(self, W, b) -> None:
        self.W = W   # (d, p)
        self.b = b   # (p, )
        self.X = None   # (n, d)
        self.dW = None
        self.dX = None
        self.db = None
    def forward(self, X):
        self.X = X   # (n, d) or (d, )
        out = np.dot(self.X, self.W) + self.b # (n, p) + (p, ) -> (n, p) broadcasting
        return out
    def backward(self, dout):
        self.dX = np.dot(dout,self.W.T)
        # ndim == 1인 경우 문제가 생긴다. 이를 해결하기 위한 조건문.
        if dout.ndim == 2: # 즉, batch로 입력되는 경우이다.
            self.dW = np.dot(self.X.T, dout)
            self.db = np.sum(dout, axis=0)  # forward시 브로드캐스팅으로 copy node 가 생략되었었다고 보고 adder를 적용
        elif dout.ndim == 1:
            self.dW = np.dot(self.X.reshape(self.X.size, 1), dout.reshape(1, dout.size))
            self.db = dout         
        else:
            print("Type or dimension is wrong.")
            raise TypeError
        return self.dX, self.dW, self.db
   
class CEE_Softmax:
    def __init__(self) -> None:
        self.y = None
        self.t = None
        self.loss = None
    def forward(self, a, t):
        self.t = t
        self.y = softmax(a)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
    def backward(self, dout):
        if dout.ndim == 2:
            batch_size = dout.shape[0]
        elif dout.ndim == 1:
            batch_size = 1
        else:
            print("Type is wrong.")
            raise TypeError
        if self.y.size == self.t.size:  # one-hot일 경우에 같다.
            da = (self.y - self.t)/batch_size
        else:
            da = self.y.copy()
            da[np.arange(batch_size), self.t] -= 1 # y 사본으로 t one-hot 만들기
            da = da / batch_size
        return da
