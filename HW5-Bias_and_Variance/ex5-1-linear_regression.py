import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.optimize as opt
import seaborn as sns

def load_data(path):
    d = sio.loadmat(path)
    return map(np.ravel, [d['X'], d['y'], d['Xval'], d['yval'], d['Xtest'], d['ytest']])

X, y, Xval, yval, Xtest, ytest = load_data('HW5-Bias_and_Variance/ex5data1.mat')

df = pd.DataFrame({'Water level': X, 'Flowing out': y})
sns.lmplot(x='Water level', y='Flowing out', data=df, fit_reg=False, height=7)
plt.show()

X, Xval, Xtest = [np.insert(x.reshape(-1, 1), 0, np.ones(x.shape[0]), axis=1) for x in (X, Xval, Xtest)]

def regularized_cost(theta, X, y, l=1):
    '''
    X: m*n
    y: m
    theta: n
    '''
    m = X.shape[0]
    inner = X @ theta.T - y
    first = inner @ inner.T / (2*m)
    
    tmp = np.array(theta)
    tmp[0] = 0
    second = tmp @ tmp.T * l / (2*m)
    
    return first + second

theta = np.ones(X.shape[1])

def regularized_gradient(theta, X, y, l=1):
    m = X.shape[0]
    tmp = np.array(theta)
    tmp[0] = 0
    ret = (X @ theta.T - y) @ X / m + l / m * tmp
    
    return ret


def linear_regression(X, y, l=1):
    theta = np.ones(X.shape[1])
    res = opt.minimize(fun=regularized_cost, x0=theta,
                      args=(X, y, l), method='TNC',
                      jac=regularized_gradient,
                      options={'disp': True})
    return res

theta = linear_regression(X, y, l=0).get('x')
# ax + b
b = theta[0] # intercept
a = theta[1] # slope
x = X[:, 1:]
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(x, y, label='Training data')
ax.plot(x, a*x+b, label='Prediction', c='r')
ax.legend(loc='best')
plt.show()

training_cost, cv_cost = [], []
m = X.shape[0]
for i in range(1, m+1):
    res = linear_regression(X[:i, :], y[:i], l=0)
    tc = regularized_cost(res.x, X[:i, :], y[:i], l=0)
    cv = regularized_cost(res.x, Xval, yval, l=0)
    
    training_cost.append(tc)
    cv_cost.append(cv)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(1, m+1), training_cost, label='Training cost')
ax.plot(np.arange(1, m+1), cv_cost, label='Cv cost')
ax.legend(loc='best')
plt.show()















