import numpy as np
from Autoencoder import NN, utils

# input data frame:
# [ BMI, Gender, Diet_Protein, Diet_Salt, Exercise, Sleep ]
# BMI: [ std:0, low:-1, high:1 ]
# Gender: [ male:0, female:1 ]
# Diet_Protein: [ std:0, low:-1, high:1 ]
# Diet_Salt: [ std:0, low:-1, high:1 ]
# Exercise(h): how many hours a day
# Sleep(h): how many hours a day

# dimension: 6
input = np.array([
    [0, 0, 0, 0, 2, 8],
    [0, 0, 1, 0, 6, 7],
    [-1, 0, 1, 1, 7, 4],
    [1, 1, 1, 1, 8, 3],
    [-1, 0, 1, -1, 7, 4],
    [1, 1, 0, 0, 8, 4],
    [-1, 0, 1, 1, 2, 7],
    [1, 0, -1, 0, 7, 6]
])


# output data frame:
# [ BUN, Creatinine]
# BUN: [ regular:1, irregular:0 ]
# Creatinine: [ regular:1, irregular:0 ]

# dimension: 2
output = np.array([
    [1, 1],
    [0, 1],
    [0, 0],
    [0, 0],
    [0, 0],
    [1, 0],
    [1, 1],
    [1, 1]
])


# build an AE
u = utils()
input_dimen = input.shape[1]
output_dimen = output.shape[1]
nodes = [input_dimen, 4, 3]
AE = utils.ae_build(u, nodes)

# train the AE
iterations = 6000
AE = utils.ae_train(u, AE, input, iterations)

# further train
nodes_f = [input_dimen, 5, 3, output_dimen]
NN = NN(nodes_f)
# put the weight of the trained AE to the new NN
for i in range(np.size(nodes) - 1):
    NN.W[i] = AE.encoders[i].W[0]
# train
iterations = 3000
for i in range(iterations):
    NN = utils.nnff(u, NN, input, output)
    NN = utils.nnff(u, NN, input, output)

# print the results
print(NN.value[-1])