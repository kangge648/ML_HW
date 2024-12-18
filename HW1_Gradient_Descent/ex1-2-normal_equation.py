import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def computeCost(X, Y, theta):
    inner = np.power((X * theta.T) - Y, 2)
    return np.sum(inner) / (2 * len(X))

def normalEqn(X, Y):
    theta = np.linalg.inv(X.T@X)@X.T@Y
    return theta

# transform theta to the origin data
# theta_0 = (theta_0 - theta_1 * mean(x_1)/ std(x_1) - ... - theta_n * mean(x_n)/ std(x_n)) * std(y) + mean(y)
# theta_1 = theta_1 * std(y) / std(x_1)
# ...
# theta_n = theta_n * std(y) / std(x_n)
def theta_transform(theta, means, stds):
    temp = means[:-1] * theta[1:] / stds[:-1]
    theta[0] = (theta[0] - np.sum(temp)) * stds[-1] + means[-1]
    theta[1:] = theta[1:] * stds[-1] / stds[:-1]
    return theta.reshape(1, -1)

# 读入数据
path = 'ex1data2.txt'
data = pd.read_csv(path, header=None, names=['Size', 'Number', 'Price'])
# print(data.head())
# print(data.describe())

# 记录数据，之后特征正则化
means = data.mean().values
stds = data.std().values
mins = data.min().values
maxs = data.max().values
data_ = data.values

# 特征正则化并插入1
data = (data - data.mean()) / data.std()
data.insert(0, 'Ones', 1)

# 建立输入输出，得到theta
cols = data.shape[1]
X = data.iloc[:, 0:cols-1]
Y = data.iloc[:, cols-1:cols]
X = np.matrix(X.values)
Y = np.matrix(Y.values)
theta = np.matrix(np.array([0, 0, 0]))
theta = normalEqn(X, Y)

# 转换theta，使之符合初始数据
theta = np.array(theta.reshape(-1, 1))
means = means.reshape(-1, 1)
stds = stds.reshape(-1, 1)
theta = theta_transform(theta, means, stds)

# 画出拟合平面
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
X_ = np.arange(mins[0], maxs[0]+1, 1)
Y_ = np.arange(mins[1], maxs[1]+1, 1)
X_, Y_ = np.meshgrid(X_, Y_)
Z_ = theta[0,0] + theta[0,1] * X_ + theta[0,2] * Y_

# 手动设置角度
ax.view_init(elev=10, azim=80)

ax.set_xlabel('Size')
ax.set_ylabel('Bedrooms')
ax.set_zlabel('Price')

ax.set_xticks(())
ax.set_yticks(())
ax.set_zticks(())
ax.plot_surface(X_, Y_, Z_, rstride=1, cstride=1, color='red')

ax.scatter(data_[:, 0], data_[:, 1], data_[:, 2])
plt.show()