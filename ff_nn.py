import numpy as np

n_x = 7             # 7 input nodes
n_h = 9             # 9 nodes in hidden layer 1
n_h2 = 15           # 15 nodes in hidden layer 2
n_y = 3             # 3 output nodes
w1_shape = [9,7]    # 7 input, 9 hidden
w2_shape = [15,9]   # 9 hidden input, 15 hidden
w3_shape =[3,15]    # 15 hidden input, 3 output


def get_weights_from_encoded(individual):
    w1 = individual[0:w1_shape[0] * w1_shape[1]]
    w2 = individual[w1_shape[0] * w1_shape[1]:w2_shape[0] * w2_shape[1] + w1_shape[0] * w1_shape[1]]
    w3 = individual[w2_shape[0] * w2_shape[1] + w1_shape[0] * w1_shape[1]:]

    return (w1.reshape(w1_shape[0], w1_shape[1])), w2.reshape(w2_shape[0], w2_shape[1]), w3.reshape(w3_shape[0], w3_shape[1])


def softmax(z):
    return np.exp(z.T) / np.sum(np.exp(z.T), axis=1).reshape(-1,1)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward_propagation(x, individual):
    w1, w2, w3 = get_weights_from_encoded(individual)

    z1 = np.matmul(w1, x.T)
    a1 = np.tanh(z1)

    z2 = np.matmul(w2, a1)
    a2 = np.tanh(z2)

    z3 = np.matmul(w3, a2)
    a3 = softmax(z3)

    return a3
