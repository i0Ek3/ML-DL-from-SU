import d2lzh as d2l
from mxnet import autograd, nd

# fetch dataset
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# initial model parameters
num_inputs, num_outputs = 784, 10
W = nd.random_normal(scale=0.01, shape=(num_inputs, num_outputs))
b = nd.zeros(num_outputs)
W.attach_grad()
b.attach_grad()

# softmax
X = nd.array([[1, 2, 3], [4, 5, 6]])
X.sum(axis=0, keepdims=True), X.sum(axis=1, keepdims=True)

def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(axis=1, keepdims=True)
    return X_exp / partition

X = nd.random_normal(shape=(2, 5))
X_prob = softmax(X)
X_prob, X_prob.sum(axis=1)

# model
def net(X):
    return softmax(nd.dot(X.reshape((-1, num_inputs)), W) + b)

# loss
y_hat = nd.array([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = nd.array([0, 2], dtype='int32')
nd.pick(y_hat, y)

def cross_entropy(y_hat, y):
    return -nd.pick(y_hat, y).log()

# accuracy
def accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()
accuracy(y_hat, y)

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        y = y.astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y).sum().asscalar()
        n += y.size
    return acc_sum / n
evaluate_accuracy(test_iter, net)

# train the model
num_epochs, lr = 5, 0.1
d2l.train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)

# prediction
for X, y in test_iter:
    break
true_labels = d2l.get_fashion_mnist_labels(y.asnumpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
d2l.show_fashion_mnist(X[0:9], titles[0:9])
