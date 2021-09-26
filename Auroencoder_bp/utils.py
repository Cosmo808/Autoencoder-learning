import numpy as np
from bean import nn, autoencoder

def aebuilder(nodes):
    layers = len(nodes)
    ae = autoencoder()
    for i in range(layers - 1):
        ae.add_one(nn([nodes[i], nodes[i + 1], nodes[i]]))
    return ae

def aetrain(ae, x, iterations):
    layers = len(ae.encoders)
    for i in range(layers):
        # 单层训练
        ae.encoders[i] = nntrain(ae.encoders[i], x, x, iterations)
        # 训练后取中间值作为下一次输入
        nntemp = nnff(ae.encoders[i], x, x)
        x = nntemp.values[1]
    return ae

def nntrain(nn, x, y, iterations):
    for i in range(iterations):
        nnff(nn, x, y)
        nnbp(nn)
    return nn

# feed forward 前馈函数
def nnff(nn, x, y):
    layers = nn.layers
    numbers = x.shape[0]
    nn.values[0] = x
    for i in range(1, layers):
        # sigmoid(w*x+b)
        nn.values[i] = sigmoid(np.dot(nn.values[i - 1], nn.W[i - 1]) + nn.B[i - 1])
    # 最后一层与实际的误差
    nn.error = y - nn.values[layers - 1]
    # 代价损失
    nn.loss = 1.0 / 2.0 * (nn.error ** 2).sum() / numbers
    return nn

# 激活函数
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# back propagation
def nnbp(nn):
    layers = nn.layers
    # 初始化delta
    delta = list()
    for i in range(layers):
        delta.append(0)
    # delta最后一层为
    delta[layers - 1] = -nn.error * nn.values[layers - 1] * (1 - nn.values[layers - 1])
    # 其他层的delta
    for j in range(1, layers - 1)[::-1]:
        delta[j] = np.dot(delta[j + 1], nn.W[j].T) * nn.values[j] * (1 - nn.values[j])
    # 更新W和B
    for k in range(layers - 1):
        nn.W[k] -= nn.u * np.dot(nn.values[k].T, delta[k + 1]) / (delta[k + 1].shape[0])
        nn.B[k] -= nn.u * delta[k + 1] / (delta[k + 1].shape[0])
    return nn