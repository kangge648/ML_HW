import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

input_size = 2  # 输入的尺寸
hidden_size = 2  # 隐含层的节点个数
class_size = 1  # 分类个数
lr = 0.9  # 学习率


# 定义两层的网络模型
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, class_size):
        super(NeuralNetwork, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, class_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # 激活函数是sigmoid函数

    def forward(self, x):
        out = self.fc(x)
        out = self.sigmoid(out)
        out = self.output(out)
        out = self.sigmoid(out)
        return out


model = NeuralNetwork(input_size, hidden_size, class_size)

# optimizer=torch.optim.Adam(model.parameters(),lr)
optimizer = torch.optim.SGD(model.parameters(), lr)
loss_func = nn.MSELoss()

# 构建训练数据集
train_x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
train_y = np.array([[0], [1], [1], [0]])

train_x = torch.tensor(train_x, dtype=torch.float)
train_y = torch.tensor(train_y, dtype=torch.float)

loss = 1000
loss_lst = []
iter_lst = []
iter_cnt = 0
while loss >= 0.008:
    iter_cnt += 1  # 迭代次数加一

    output = model(train_x)  # 代入模型计算
    loss = loss_func(output, train_y)
    loss_lst.append(loss.detach().numpy())
    iter_lst.append(iter_cnt)
    print("loss:", loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # for param_tensor in model.state_dict():
    # print(param_tensor,model.state_dict()[param_tensor])

print("total loss:", loss)
print("total iter:", iter_cnt)

print("result:")
print(model(torch.tensor([0, 0], dtype=torch.float)))
print(model(torch.tensor([0, 1], dtype=torch.float)))
print(model(torch.tensor([1, 0], dtype=torch.float)))
print(model(torch.tensor([1, 1], dtype=torch.float)))

plt.title('Learning curve', fontsize=20)
plt.plot(iter_lst, loss_lst, '.-')
plt.xlabel('iteration', fontsize=20)
plt.ylabel('Train loss', fontsize=20)
plt.grid()
plt.show()
# plt.savefig('fig/fig1.png')
