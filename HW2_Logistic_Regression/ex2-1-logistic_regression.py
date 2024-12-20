import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# sigmoid function, z is a scalar
def Sigmoid(z):
    return 1/(1 + np.exp(-z))

# nums = np.arange(-10, 10, step=0.5)
# fig, ax = plt.subplots(figsize=(12, 8))
# ax.plot(nums, sigmoid(nums), 'r')
# plt.show()

# compute the loss
# input (dimensions): X is m, n+1; Y is m, 1; theta is n+1, 1
# output: a scalar
def computeCost(theta, X, Y):
    inner = np.sum(- Y.T @ np.log(Sigmoid(X @ theta)) - (1 - Y.T) @ np.log(1 - Sigmoid(X @ theta)))
    return inner / len(X)

# gradient desent
# input (dimensions): X is m, n+1; Y is m, 1; theta is n+1, 1; alpha and iteration are scalars
# output: theta is n+1, 1; lossses
def gradient_descent(X, Y, theta, alpha, iteration, losses):
    Y = Y.reshape(-1, 1)
    for i in range(iteration):
        loss = computeCost(theta, X, Y)
        losses.append(loss)
        gradient = 1/len(X) * X.T @ (Sigmoid(X @ theta) - Y)
        theta -= alpha * gradient

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

# 建立输入输出
data.insert(0, 'Ones', 1)
cols = data.shape[1]
X = data.iloc[:, 0:-1].values
Y = data.iloc[:, -1].values
theta = np.array([[0.0], [0.0], [0.0]])

# # 只用自己编写的梯度下降法，需要alpha很小，而iteration很大才行
# # 也可以选用内置优化方法
# # --------------------------自行编写求解-------------------------
# alpha = 0.001
# losses = []
# iteration = 100000
# theta, losses = gradient_descent(X, Y, theta, alpha, iteration, losses)
# print(theta)

# plt.figure(figsize=(12, 8))
# plt.xlabel('Iteration')
# plt.ylabel('Cost funtcion')
# plt.plot(range(iteration), losses, color='blue')
# plt.title('Cost Function by Iteration')
# plt.show()
# # --------------------------自行编写求解-------------------------


# --------------------------内置优化函数-------------------------
def gradient(theta, X, Y):
    return 1/len(X) * X.T @ (Sigmoid(X @ theta) - Y)

import scipy.optimize as opt
res = opt.minimize(fun=computeCost, x0=np.array(theta), args=(X, np.array(Y)), method='Newton-CG', jac=gradient)
theta = res.x
print(theta)
# --------------------------内置优化函数-------------------------

# 验证
def predict(theta, X):
    probability = Sigmoid(X @ theta)
    return [1 if x >= 0.5 else 0 for x in probability]

predictions = predict(theta, X)
correct = [1 if a^b == 0 else 0 for (a,b) in zip(predictions, Y)]
accuracy = (sum(correct) / len(correct))
print('accuracy = {0:.0f}%'.format(accuracy*100))

# 决策边界
coef = -theta / theta[2]
x = np.arange(30, 100, 0.5)
y = coef[0] + coef[1] * x

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['Score1'], positive['Score2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Score1'], negative['Score2'], s=50, c='r', marker='x', label='Not Admitted')
ax.plot(x, y, label='Decision Boundary', c='grey')
ax.legend()
ax.set_xlabel('Exam1 Score')
ax.set_ylabel('Exam2 Score')
plt.show()