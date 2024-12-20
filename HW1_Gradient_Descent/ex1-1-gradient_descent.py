# Linear regression by the gradient descent (single variable)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# compute the loss
# input (dimensions): X is m, 2; Y is m, 1; theta is 2, 1
# output: a scalar
def computeCost(X, Y, theta):
    inner = np.power((X @ theta) - Y, 2)
    return np.sum(inner) / (2 * len(X))

# gradient desent
# input (dimensions): X is m, 2; Y is m, 1; theta is 2, 1; alpha and iteration are scalars
# output: theta is 2, 1; lossses
def gradient_descent(X, Y, theta, alpha, iteration):
    losses = []
    for i in range(iteration):
        loss = computeCost(X, Y, theta)
        losses.append(loss)
        gradient = 1/len(X) * ((X @ theta) - Y).T @ X
        theta -= alpha * gradient.T

    return theta, losses

# 读入数据
path = 'HW1_Gradient_Descent/ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
# print(data.head())
# print(data.describe())

data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
plt.show()

# 新增截距项
data.insert(0, 'Ones', 1)
data.head()

cols = data.shape[1]
X = data.iloc[:, 1:cols-1]
# X = (X - X.mean()) / X.std()
X.insert(0, 'Ones', 1)
Y = data.iloc[:, cols-1:cols]
# Y = (Y - Y.mean()) / Y.std()
X = np.matrix(X.values)
Y = np.matrix(Y.values)


alpha = 0.01
iteration = 1000
losses = []
theta = np.array([[0.0], [0.0]])
theta, losses = gradient_descent(X, Y, theta, alpha, iteration)
print(theta)

# 画出拟合图像
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = theta[0,0] + theta[1,0] * x


# 归一化后得到的参数theta不能直接应用于原数据
# 给定一个新数据x，得到的对应输出的步骤：
# 1. newx = (x-mean(X)) / std(X)
# 2. y = theta[0, 0] + theta[0, 1] * newx
# 3. truey = y * std(Y) + mean (Y)
plt.figure(figsize=(12, 8))
plt.xlabel('Population')
plt.ylabel('Profit')
x_vals = np.array(data['Population']).reshape(-1, 1)  # 确保x_vals是二维的
x_vals_with_intercept = np.hstack([np.ones((x_vals.shape[0], 1)), x_vals])  # 添加截距项
predictions = x_vals_with_intercept @ theta  # 计算预测值
plt.plot(data['Population'], predictions, label='Prediction', color='red')
plt.scatter(data['Population'], data['Profit'], label='Training Data')
plt.legend(loc='best')
plt.title('Predicted Profit vs Population Size')
plt.show()

plt.figure(figsize=(12, 8))
plt.xlabel('Iteration')
plt.ylabel('Cost funtcion')
plt.plot(range(iteration), losses, color='blue')
plt.title('Cost Function by Iteration')
plt.show()