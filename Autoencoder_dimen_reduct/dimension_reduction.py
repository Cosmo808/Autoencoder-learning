import torch.nn as nn
from torch.autograd import Variable as V
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import numpy
import matplotlib.pyplot as plt
from Autoencoder import autoencoder

# 读入数据
iris = load_iris()
x = iris.data
y = iris.target
Y = y  # 在画图中备用

# 对输入进行归一化，因为autoencoder只用到了input
MMScaler = MinMaxScaler()
x = MMScaler.fit_transform(x)
iforestX = x

# 输入数据转换成神经网络接受的dataset类型，batch设定为10
tensor_x = torch.from_numpy(x.astype(numpy.float32))
tensor_y = torch.from_numpy(y.astype(numpy.float32))
my_dataset = TensorDataset(tensor_x, tensor_y)
my_dataset_loader = DataLoader(my_dataset, batch_size = 10, shuffle = False)

print(isinstance(my_dataset,Dataset))

model = autoencoder()

# 定义损失函数

criterion = nn.MSELoss()

# 定义优化函数
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)  # 如果采用SGD的话，收敛不下降

# epoch 设定为300

for epoch in range(300):
    total_loss = 0
    for i, (x, y) in enumerate(my_dataset_loader):
        _, pred = model(V(x))
        loss = criterion(pred, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss
    if epoch % 100 == 0:
        print(total_loss.data.numpy())

# 基于训练好的model做降维并可视化

x_ = []
y_ = []
for i, (x, y) in enumerate(my_dataset):
    _, pred = model(V(x))
    #loss = criterion(pred, x)
    dimension = _.data.numpy()
    x_.append(dimension[0])
    y_.append(dimension[1])

plt.scatter(numpy.array(x_), numpy.array(y_), c = Y)

for i in range(len(numpy.array(x_))):
    plt.annotate(i, (x_[i], y_[i]))

plt.show()
