import numpy as np
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt

class Network_2L:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01) -> None:
        self.params = {}
        self.params['W1'] = weight_init_std*np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std*np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = self.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = self.softmax(a2)

        return y

    def loss(self, x, t):
        y = self.predict(x)
        return self.cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        return np.sum(y == t)/float(x.shape[0])

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        
        grads['W1'] = self.num_gradient(loss_W, self.params['W1'])
        grads['b1'] = self.num_gradient(loss_W, self.params['b1'])
        grads['W2'] = self.num_gradient(loss_W, self.params['W2'])
        grads['b2'] = self.num_gradient(loss_W, self.params['b2'])

        return grads


    def sigmoid(self, a):
        return 1/(1+np.exp(-a))

    def softmax(self, a):
        C = np.max(a, axis=0)
        exponential = np.exp(a -C)
        return exponential/np.sum(exponential)

    # CEE for both one-hot encoding and valued labels
    def cross_entropy_error(self, y, t):
        """CEE:
        Cross Entropy Error for a batch y and one-hot label t.
        The idea is that one-hot encoding has only 1 at the true element and 0 elsewhere,
        so that t * log(y) == 0. Hence, the only valid element is the 1 and to 
        obtain the result, one only has to simply calculate on that element.

        parameters
        ----------
        y, t: probability array
            If y -> 0, log(y) -> -oo. Hence, delta was introduced.
        batch_size: number of data of a batch

        returns
        -------
        float
        """
        if y.ndim == 1:
            y = y.reshape(1, y.size)
            t = t.reshape(1, t.size) 

        if y.size == t.size: # if one-hot
            t = t.argmax(axis=1) # recover label values 

        delta = 1e-7
        batch_size = y.shape[0]
        return - np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size

    def num_gradient_without_batch(self, f, x):
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

        
    def num_gradient(self, f, X):
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
            return self.num_gradient_without_batch(f, X)    
        grad = np.zeros_like(X)

        for i, x in enumerate(X):    # for each record x in a batch X
            grad[i] = self.num_gradient_without_batch(f, x)

        return grad

# MNIST 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 2층 신경망 클래스
network = Network_2L(input_size=784, hidden_size=50, output_size=10)

# 하이퍼파라미터
iters_num = 20  # 반복 횟수
train_size = x_train.shape[0]
batch_size = 100      # 미니배치 크기
learning_rate = 0.2   # 학습률

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    print(i, end=" ")
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 기울기 계산
    grad = network.numerical_gradient(x_batch, t_batch)
    
    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    # 1에폭당 정확도 계산
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# 정확도 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()