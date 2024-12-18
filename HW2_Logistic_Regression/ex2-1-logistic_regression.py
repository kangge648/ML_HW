import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def Sigmoid(z):
    return 1/(1 + np.exp(-z))

# nums = np.arange(-10, 10, step=0.5)
# fig, ax = plt.subplots(figsize=(12, 8))
# ax.plot(nums, sigmoid(nums), 'r')
# plt.show()

def computeCost(X, Y, theta, i):
    # if i == 1:
    #     print(Y.T)
    #     print(np.log(Sigmoid(X @ theta.T)))
    inner = - Y.T @ np.log(Sigmoid(X @ theta.T)) - (1 - Y.T) @ np.log(1 - Sigmoid(X @ theta.T))
    return inner / len(X)

def gradient_descent(X, Y, theta, alpha, iteration, losses):
    for i in range(iteration):
        Y_pred = Sigmoid(X @ theta.T)
        loss = computeCost(X, Y, theta, i)
        losses.append(loss)
        gradient = 1/len(X) * X.T * (Y_pred - Y)
        theta -= alpha * gradient.T

    return theta, losses

# 读入数据
path = 'HW2_Logistic_Regression/ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Score1', 'Score2', 'Ispermitted'])
# print(data.head())
# print(data.describe())
positive = data[data['Ispermitted'].isin([1])]
negative = data[data['Ispermitted'].isin([0])]

# fig, ax = plt.subplots(figsize=(12, 8))
# ax.scatter(positive['Score1'], positive['Score2'], s=50, c='b', marker='o', label='permitted')
# ax.scatter(negative['Score1'], negative['Score2'], s=50, c='r', marker='x', label='Not permitted')
# ax.legend()
# ax.set_xlabel('Exam1 Score')
# ax.set_ylabel('Exam2 Score')
# plt.show()

# 建立输入输出，得到theta
data.insert(0, 'Ones', 1)
cols = data.shape[1]
X = data.iloc[:, 0:cols-1]
Y = data.iloc[:, cols-1:cols]
X = np.matrix(X.values)
Y = np.matrix(Y.values)
theta = np.matrix(np.array([0.0, 0.0, 0.0]))
alpha = 0.01
losses = []
iteration = 1000
theta, losses = gradient_descent(X, Y, theta, alpha, iteration, losses)
print(type(losses))

# plt.figure(figsize=(12, 8))
# plt.xlabel('Iteration')
# plt.ylabel('Cost funtcion')
# plt.plot(range(iteration), losses, color='blue')
# plt.title('Cost Function by Iteration')
# plt.show()