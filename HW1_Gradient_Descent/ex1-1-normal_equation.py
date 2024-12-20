# Linear regression by the normal equation (single variable)
# Normal equation: theta = (X.T * X)^(-1) * X.T * Y

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# compute the loss
# input (dimensions): X is m, 2; Y is m, 1; theta is 2, 1
# output: a scalar
def computeCost(X, Y, theta):
    inner = np.power((X @ theta) - Y, 2)
    return np.sum(inner) / (2 * len(X))

# normal equation
# input (dimensions):  X is m, 2; Y is m, 1; theta is 2, 1 
# output (dimensions): theta is 2, 1
def normalEqn(X, Y):
    theta = np.linalg.inv(X.T @ X) @ X.T @ Y
    return theta

# 读入数据
path = 'HW1_Gradient_Descent/ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
# print(data.head())
# print(data.describe())

data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
plt.show()

# 新增截距项
data.insert(0, 'Ones', 1)

cols = data.shape[1]
X = data.iloc[:, 0:cols-1]
Y = data.iloc[:, cols-1:cols]

X = np.matrix(X.values)
Y = np.matrix(Y.values)
theta = np.matrix(np.array([0, 0]))
theta = normalEqn(X, Y)
print(theta)
computeCost(X, Y, theta)

# 画出拟合图像
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = theta[0,0] + theta[1,0] * x


plt.figure(figsize=(12, 8))
plt.xlabel('Population')
plt.ylabel('Profit')
l1 = plt.plot(x, f, label='Prediction', color='red')
l2 = plt.scatter(data.Population, data.Profit, label='Traing Data', )
plt.legend(loc='best')
plt.title('Predicted Profit vs Population Size')
plt.show()