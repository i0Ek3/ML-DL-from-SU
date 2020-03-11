import d2lzh as d2l
from mxnet import autograd, nd
from mxnet.gluon import loss as gl

# fetch dataset
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# define model parameters
num_inputs, num_hiddens, num_outputs = 784, 256, 10
w1 = nd.random_normal(scale=0.01, shape=(num_inputs, num_hiddens))
b1 = nd.zeros(num_hiddens)
w2 = nd.random_normal(scale=0.01, shape=(num_hiddens, num_hiddens))
b2 = nd.zeros(num_hiddens)
w3 = nd.random_normal(scale=0.01, shape=(num_hiddens, num_outputs))
b3 = nd.zeros(num_outputs)
params = [w1, b1, w2, b2, w3, b3]

for p in params:
    p.attach_grad()

# activation function
def relu(X):
    return nd.maximum(X, 0)

# model
def net(X):
    X = X.reshape((-1, num_inputs))
    H1 = relu(nd.dot(X, w1) + b1)
    H2 = relu(nd.dot(H1, w2) + b2)
    return nd.dot(H2, w3) + b3

# loss function
loss = gl.SoftmaxCrossEntropyLoss()

# train the model
num_epochs, lr = 5, 0.1
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)
