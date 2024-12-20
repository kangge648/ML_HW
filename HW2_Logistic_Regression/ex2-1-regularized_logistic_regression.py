# logisitic regression with regularization

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt


# Sigmoid function, z is a scalar
def Sigmoid(z):
    return 1/(1 + np.exp(-z))

# compute the loss
# input (dimensions): X is m, n+1; Y is m, 1; theta is n+1, 1
# output: a scalar
def cost(theta, X, Y):
    inner = np.sum(- Y.T @ np.log(Sigmoid(X @ theta)) - (1 - Y.T) @ np.log(1 - Sigmoid(X @ theta)))
    return inner / len(X)

def regularized_cost(theta, X, Y, lam=1):
    theta_1n = theta[1:]
    regularized_term = lam / (2 * len(X)) * np.power(theta_1n, 2).sum()
    return cost(theta, X, Y) + regularized_term

# gradient desent
# input (dimensions): X is m, n+1; Y is m, 1; theta is n+1, 1; alpha and iteration are scalars
# output: theta is n+1, 1; lossses
def gradient_descent(X, Y, theta, lam, alpha, iteration, losses):
    Y = Y.reshape(-1, 1)
    for i in range(iteration):
        loss = cost(theta, X, Y)
        losses.append(loss)
        gradient = 1/len(X) * X.T @ (Sigmoid(X @ theta) - Y) + lam / len(X) * theta
        gradient[0] -= lam / len(X) * theta[0]
        theta -= alpha * gradient

    return theta, losses

# get new features
# input: x is m, 1; y is m, 1; power is a scalar; should output as array or pd.series
# return a new and extend feature matrix
def feature_mapping(x, y, power, as_ndarray=False):
    data = {'f{0}{1}'.format(i-p, p): np.power(x, i-p) * np.power(y, p)
                for i in range(0, power+1)
                for p in range(0, i+1)
           }
    if as_ndarray:
        return pd.DataFrame(data).values
    else:
        return pd.DataFrame(data)

# get the input
path = 'HW2_Logistic_Regression/ex2data2.txt'
df = pd.read_csv(path, header=None, names=['Test1', 'Test2', 'Accepted'])
# df.head()
positive = df[df['Accepted'].isin([1])]
negative = df[df['Accepted'].isin([0])]
# fig, ax = plt.subplots(figsize=(12, 8))
# ax.scatter(positive['Test1'], positive['Test2'], s=50, c='b', marker='o', label='Accepted')
# ax.scatter(negative['Test1'], negative['Test2'], s=50, c='r', marker='x', label='Rejected')
# ax.legend()
# ax.set_xlabel('Test1 Score')
# ax.set_ylabel('Test2 Score')
# plt.show()

# 构建新的特征矩阵，为data
x1 = df.Test1.values
x2 = df.Test2.values
Y = df.Accepted.values
data = feature_mapping(x1, x2, power=6)
theta = np.zeros(data.shape[1])
theta = theta.reshape(-1, 1)
X = feature_mapping(x1, x2, power=6, as_ndarray=True)

# # 只用自己编写的梯度下降法，需要alpha很小，而iteration很大才行
# # 也可以选用内置优化方法
# # --------------------------自行编写求解-------------------------
# alpha = 0.01
# losses = []
# iteration = 20000
# lam = 1
# theta, losses = gradient_descent(X, Y, theta, lam, alpha, iteration, losses)
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
    return (1/len(X) * X.T @ (Sigmoid(X @ theta.T) - Y))

def regularized_gradient(theta, X, Y, lam=1):
    theta_1n = theta[1:]
    regularized_theta = lam / len(X) * theta_1n
    regularized_term = np.concatenate([np.array([0]), regularized_theta])
    
    return  gradient(theta, X, Y) + regularized_term 
    
res = opt.minimize(fun=regularized_cost, x0=theta, args=(X, Y), method='Newton-CG', jac=regularized_gradient)
print(res)
# --------------------------内置优化函数-------------------------

def predict(theta, X):
    probability = Sigmoid(X @ theta.T)
    return probability >= 0.5

# from sklearn.metrics import classification_report
# Y_pred = predict(res.x, X)
# print(classification_report(Y, Y_pred))


# --------------------------过拟合分析-------------------------
# 得到theta
def find_theta(power, lam):
    path = 'HW2_Logistic_Regression/ex2data2.txt'
    df = pd.read_csv(path, header=None, names=['Test1', 'Test2', 'Accepted'])
    Y = df.Accepted
    x1 = df.Test1.values
    x2 = df.Test2.values
    X = feature_mapping(x1, x2, power, as_ndarray=True)
    theta = np.zeros(X.shape[1])
    res = opt.minimize(fun=regularized_cost, x0=theta, args=(X, Y, lam), method='TNC', jac=regularized_gradient)
    return res.x

# 决策边界，thetaX = 0, thetaX <= threshhold
def find_decision_boundary(density, power, theta, threshhold):
    t1 = np.linspace(-1, 1.2, density)
    t2 = np.linspace(-1, 1.2, density)
    cordinates = [(x, y) for x in t1 for y in t2]
    x_cord, y_cord = zip(*cordinates)
    mapped_cord = feature_mapping(x_cord, y_cord, power)
    
    pred = mapped_cord.values @ theta.T
    decision = mapped_cord[np.abs(pred) <= threshhold]
    
    return decision.f10, decision.f01

# 画决策边界
def draw_boundary(power, lam):
    density = 1000
    threshhold = 2 * 10**-3
    
    theta = find_theta(power, lam)
    x, y = find_decision_boundary(density, power, theta, threshhold)
    positive = df[df['Accepted'].isin([1])]
    negative = df[df['Accepted'].isin([0])]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(positive['Test1'], positive['Test2'], s=50, c='b', marker='o', label='Accepted')
    ax.scatter(negative['Test1'], negative['Test2'], s=50, c='g', marker='x', label='Rejected')
    ax.scatter(x, y, s=50, c='r', marker='.', label='Decision Boundary')
    ax.legend()
    ax.set_xlabel('Test1 Score')
    ax.set_ylabel('Test2 Score')

    plt.show()
# --------------------------过拟合分析-------------------------

draw_boundary(6, lam=1)
# draw_boundary(6, lam=0)
# draw_boundary(6, lam=100)