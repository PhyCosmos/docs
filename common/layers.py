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
        dx = dout
        return dx

class Sigmoid:
    def __init__(self) -> None:
        self.out = None
    def forward(self, x):
        self.out = sigmoid(x)
        return self.out
    def backward(self, dout):
        return dout * (1.0 - self.out) * self.out

# class Affine:
#     def __init__(self, W, b) -> None:
#         self.W = W   # (d, p)
#         self.b = b   # (p, )
#         self.X = None   # (n, d)
#         self.dW = None
#         self.dX = None
#         self.db = None
#     def forward(self, X):
#         self.X = X   # (n, d) or (d, )
#         out = np.dot(self.X, self.W) + self.b # (n, p) + (p, ) -> (n, p) broadcasting
#         return out
#     def backward(self, dout):
#         self.dX = np.dot(dout,self.W.T)
#         # ndim == 1인 경우 문제가 생긴다. 이를 해결하기 위한 조건문.
#         if dout.ndim == 2: # 즉, batch로 입력되는 경우이다.
#             self.dW = np.dot(self.X.T, dout)
#             self.db = np.sum(dout, axis=0)  # forward시 브로드캐스팅으로 copy node 가 생략되었었다고 보고 adder를 적용
#         elif dout.ndim == 1:
#             self.dW = np.dot(self.X.reshape(self.X.size, 1), dout.reshape(1, dout.size))
#             self.db = dout         
#         else:
#             print("Type or dimension is wrong.")
#             raise TypeError
#         return self.dX
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        # 가중치와 편향 매개변수의 미분
        self.dW = None
        self.db = None

    def forward(self, x):
        """ 해당 layer를 거쳐 내보내는 값을 결정한다. 여기서는
        Affine 계산값. backward 계산에 활용하기 위해 초기 shape을 
        클래스 변수로 기억시켜 둔다.        

        Args:
            x ((d,..,f) shaped ndarray): 반드시 2-dim 이상의 입력값 ndarray로서
                shape의 첫 인덱스는 data record를 지칭해야 한다. 즉, d 개의
                데이터(샘플)를 입력한다는 의미가 된다.

        Returns:
            (d,r) shaped ndarray: 계산 결과를 반환한다. 결과적인 shape은 클래스
                마다 다르다.
        """
        # 텐서 대응        
        self.original_x_shape = x.shape
        # One shape dimension can be -1. In this case, 
        # the value is inferred from the length of the array 
        # and remaining dimensions. e.g. (2, 3, 4) -> (2, 12)
        x = x.reshape(x.shape[0], -1) # (1,f)->(1,f), (d,f)->(d,f), (d,m,f)->(d,m*f)...
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.mean(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 입력 데이터 모양 변경(텐서 대응)
        return dx  
     
class SoftmaxWithLoss:
    def __init__(self) -> None:
        self.y = None
        self.t = None
        self.loss = None
    def forward(self, a, t):
        self.t = t
        self.y = softmax(a)
        self.loss = cross_entropy_error(self.y, self.t) # ndarray가 아닌 batch의 **평균** loss값(float)
        return self.loss
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 정답 레이블이 원-핫 인코딩 형태일 때
            dx = (self.y - self.t)/batch_size # sum으로 브로드캐스팅 grad를 구할 때/ batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx/batch_size # sum으로 브로드캐스팅 grad를 구할 때/ batch_size
        
        return dx
    
class BatchNormalization:
    """각 층의 활성화값이 적당히 분포되도록 조정하는 layer이다.
    """
    def __init__(self, gamma, beta, momentum=0.9, 
                 running_mean=None, running_var=None) -> None:
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None # 합성곱계층은 4차원, 완전연결계층은 2차원
        
        # test에 사용할 평균과 분산
        self.running_mean = running_mean
        self.running_var = running_var
        
        # backward 에서 사용할 중간 데이터
        self.batch_size = None
        self.ac = None
        self.var = None
        self.std = None
        self.an = None
        self.dgamma = None
        self.dbeta = None
        
    def forward(self, a, train_flg=True):
        self.input_shape = a.shape
        if a.ndim !=2:
            N, C, H, W = x.shape
            a = a.reshape(N, -1)
        
        out = self.__forward(a, train_flg)
        
        return out.reshape(*self.input_shape)
    
    def __forward(self, a, train_flg=True):
        if self.running_mean is None:
            N, D = a.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
            
        if train_flg:
            mu = a.mean(axis=0)
            ac = a - mu
            var = np.mean(ac**2, axis=0)
            std = np.sqrt(var + 10e-7)
            an = ac / std
            
            self.batch_size = a.shape[0]
            self.ac = ac
            self.var = var
            self.std = std
            self.an = an
            self.running_mean = self.momentum*self.running_mean + (1-self.momentum)*mu
            self.running_var = self.momentum*self.running_var + (1-self.momentum)*var
        else:
            ac = a - self.running_mean
            an = ac / (np.sqrt(self.running_var + 10e-7))
            
        out = self.gamma * an + self.beta
        return out
    
    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)
            
        da = self.__backward(dout)
         
        da = da.reshape(*self.input_shape)
        return da
        
    def __backward(self, dout):
        dbeta = dout.mean(axis=0)
        dgamma = np.mean(self.an*dout, axis=0)
        dan = self.gamma * dout
        da = (1-1/self.batch_size)/self.std* \
            (1-1/self.batch_size/self.var*self.ac)*dan 
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return da
# class BatchNormalization:
#     """
#     http://arxiv.org/abs/1502.03167
#     """
#     def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
#         self.gamma = gamma
#         self.beta = beta
#         self.momentum = momentum
#         self.input_shape = None # 합성곱 계층은 4차원, 완전연결 계층은 2차원  

#         # 시험할 때 사용할 평균과 분산
#         self.running_mean = running_mean
#         self.running_var = running_var  
        
#         # backward 시에 사용할 중간 데이터
#         self.batch_size = None
#         self.xc = None
#         self.xn = None
#         self.std = None
#         self.dgamma = None
#         self.dbeta = None

#     def forward(self, x, train_flg=True):
#         self.input_shape = x.shape
#         if x.ndim != 2:
#             N, C, H, W = x.shape
#             x = x.reshape(N, -1)

#         out = self.__forward(x, train_flg)
        
#         return out.reshape(*self.input_shape)
            
#     def __forward(self, x, train_flg):
#         if self.running_mean is None:
#             N, D = x.shape
#             self.running_mean = np.zeros(D)
#             self.running_var = np.zeros(D)
                        
#         if train_flg:
#             mu = x.mean(axis=0)
#             xc = x - mu
#             var = np.mean(xc**2, axis=0)
#             std = np.sqrt(var + 10e-7)
#             xn = xc / std
            
#             self.batch_size = x.shape[0]
#             self.xc = xc
#             self.xn = xn
#             self.std = std
#             self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
#             self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
#         else:
#             xc = x - self.running_mean
#             xn = xc / ((np.sqrt(self.running_var + 10e-7)))
            
#         out = self.gamma * xn + self.beta 
#         return out

#     def backward(self, dout):
#         if dout.ndim != 2:
#             N, C, H, W = dout.shape
#             dout = dout.reshape(N, -1)

#         dx = self.__backward(dout)

#         dx = dx.reshape(*self.input_shape)
#         return dx

#     def __backward(self, dout):
#         dbeta = dout.sum(axis=0)
#         dgamma = np.sum(self.xn * dout, axis=0)
#         dxn = self.gamma * dout
#         dxc = dxn / self.std
#         dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
#         dvar = 0.5 *        dstd / self.std
#         dxc += (2.0 / self.batch_size) * self.xc * dvar
#         dmu = np.sum(dxc, axis=0)
#         dx = dxc - dmu / self.batch_size
        
#         self.dgamma = dgamma
#         self.dbeta = dbeta
        
#         return dx

class Dropout:
    def __init__(self, dropout_ratio=0.5) -> None:
        self.dropout_ratio = dropout_ratio
        self.mask = None
        
    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(**x.shape) > self.dropout_ratio
            return x*self.mask
        else:
            return x*(1.0-self.dropout_ratio)
        
    def backward(self, dout):
        return dout*self.mask
        