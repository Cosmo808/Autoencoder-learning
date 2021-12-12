import numpy as np


# neural network
class NN:
    def __init__(self, nodes):
        self.layers = len(nodes)
        self.nodes = nodes
        self.u = 1.0  # learning rate
        self.W = list()  # weight
        self.B = list()  # bias
        self.value = list()
        self.error = 0
        self.loss = 0
        # initialization
        for i in range(self.layers - 1):
            # generate a random matrix of -0.5 ~ 0.5 and append it to weight
            self.W.append(np.random.random((self.nodes[i], self.nodes[i + 1])) - 0.5)
            # initial bias
            self.B.append(0)
        for i in range(self.layers):
            # initial value
            self.value.append(0)


# AE
class autoencoder:
    def __init__(self):
        self.encoders = list()

    def ae_append(self, nn):
        self.encoders.append(nn)


# functions of AE
class utils:
    def ae_build(self, nodes):
        ae = autoencoder()
        layers = len(nodes)
        for i in range(layers - 1):
            nn = NN([nodes[i], nodes[i + 1], nodes[i]])
            ae.ae_append(nn)
        return ae

    def ae_train(self, ae, input, iterations):
        layers = len(ae.encoders)
        # train every layer of AE
        for i in range(layers):
            nn = ae.encoders[i]
            # ff and bp for iterations
            for j in range(iterations):
                # input and out are the same in AE
                nn = self.nnff(nn, input, input)
                nn = self.nnbp(nn)
            ae.encoders[i] = nn
            # change the input as the value of the median layer
            input = self.nnff(nn, input, input).value[1]
        return ae

    def nnff(self, nn, input, output):
        layers = nn.layers
        nn.value[0] = input
        for i in range(1, layers):
            # sigmoid(w*x+b)
            nn.value[i] = self.sigmoid(np.dot(nn.value[i - 1], nn.W[i - 1]) + nn.B[i - 1])

        # calculate error
        nn.error = output - nn.value[layers - 1]
        # calculate loss
        nn.loss = 1.0 / 2.0 * (nn.error ** 2).sum() / np.size(input)
        return nn

    def nnbp(self, nn):
        layers = nn.layers
        # initial delta
        delta = list()
        for i in range(layers):
            delta.append(0)
        # last layer of delta
        delta[layers - 1] = -nn.error * nn.value[layers - 1] * (1 - nn.value[layers - 1])
        # other layers of delta
        for j in range(1, layers - 1)[::-1]:
            delta[j] = np.dot(delta[j + 1], nn.W[j].T) * nn.value[j] * (1 - nn.value[j])
        # update weight and bias
        for k in range(layers - 1):
            nn.W[k] -= nn.u * np.dot(nn.value[k].T, delta[k + 1]) / (delta[k + 1].shape[0])
            nn.B[k] -= nn.u * delta[k + 1] / (delta[k + 1].shape[0])
        return nn

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
