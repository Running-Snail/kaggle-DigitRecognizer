import numpy as np


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


def sigmoid_gradient(x):
    g = sigmoid(x)
    return g*(1-g)


def extend_1(arr):
    return np.concatenate(([1.0], arr))


class BPNN(object):
    def __init__(self, input_layer, hidden_layer, output_layer):
        super(BPNN, self).__init__()
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.random_start()

    def feedforward(self, x):
        a1 = extend_1(x)
        z2 = np.dot(a1, self.theta1)
        a2 = extend_1(sigmoid(z2))
        z3 = np.dot(a2, self.theta2)
        a3 = sigmoid(z3)
        return a1, z2, a2, z3, a3

    def backpropagation(self, x, y):
        m = x.shape[0]
        K = y.shape[1]
        for i in xrange(m):
            xi = x[i]
            yi = y[i]
            a1, z2, a2, z3, a3 = self.feedforward(xi)
            err3 = a3 - yi # 1x10
            # err2 = self.theta2 * err3 * sigmoid_gradient(z2) # 26x10 * 1x10 * 

    def cost(self, x, y, regularized=True, lamb=1.0):
        m = x.shape[0]
        K = y.shape[1]
        J = 0
        for i in xrange(m):
            _, _, _, _, a3 = self.feedforward(x[i])
            for k in xrange(K):
                J = J - y[i][k]*np.log(a3[k]) - (1-y[i][k])*np.log(1-a3[k])
        J = J/m
        if regularized:
            theta1_t = np.copy(self.theta1)
            theta1_t[0] = 0.0
            theta2_t = np.copy(self.theta2)
            theta2_t[0] = 0.0
            J += lamb*(np.sum(theta1_t*theta1_t) + np.sum(theta2_t*theta2_t))/(m+m)
        return J

    def epsilon_init(self, nin, nout):
        return np.sqrt(6.0)/np.sqrt(nin + nout)

    def random_start(self):
        # set random start of theta
        epsilon1 = self.epsilon_init(self.input_layer, self.hidden_layer)
        epsilon2 = self.epsilon_init(self.hidden_layer, self.output_layer)
        self.theta1 = np.random.rand(self.input_layer+1, self.hidden_layer) * 2 * epsilon1 - epsilon1
        self.theta2 = np.random.rand(self.hidden_layer+1, self.output_layer) * 2 * epsilon2 - epsilon2
