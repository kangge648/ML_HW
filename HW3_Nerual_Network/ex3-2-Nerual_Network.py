import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.io import loadmat
from sklearn.metrics import classification_report

# X and y denote the features and labels
def load_data(path, transpose=True):
    data = loadmat(path)
    X = data['X']
    y = data['y']
    y = y.reshape(y.shape[0])
    if transpose:
        X = np.array([im.reshape((20,20)).T.reshape(400) for im in X])
    return X, y


def load_weight(path):
    data = loadmat(path)
    return data['Theta1'], data['Theta2']


def sigmoid(z):
    return 1 / (1 + np.exp(-z))



# 输入
theta1, theta2 = load_weight('HW3_Nerual_Network/ex3weights.mat')
X, y = load_data('HW3_Nerual_Network/ex3data1.mat', transpose=False)
X = np.insert(X, 0, np.ones(X.shape[0]), axis=1) # 插入截距项
a1 = X

# 输入线性化及加入激活函数
z2 = a1 @ theta1.T
z2 = np.insert(z2, 0, np.ones(z2.shape[0]), axis=1) # 插入截距项
a2 = sigmoid(z2)

# 输入线性化及加入激活函数
z3 = a2 @ theta2.T
a3 = sigmoid(z3)
print(a3.shape)

y_pred = np.argmax(a3, axis=1)+1
print(classification_report(y, y_pred))