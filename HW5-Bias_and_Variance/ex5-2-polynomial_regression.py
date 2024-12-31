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
                      options={'disp': False})
    return res


training_cost, cv_cost = [], []
m = X.shape[0]
for i in range(1, m+1):
    res = linear_regression(X[:i, :], y[:i], l=0)
    tc = regularized_cost(res.x, X[:i, :], y[:i], l=0)
    cv = regularized_cost(res.x, Xval, yval, l=0)
    
    training_cost.append(tc)
    cv_cost.append(cv)

def normalize_feature(df):
    return df.apply(lambda col: (col - col.mean()) / col.std())
    # df = (df - df.mean()) / data.std()

def poly_features(x, power, as_ndarray=False):
    data = {'f{}'.format(i): np.power(x, i) for i in range(1, power+1)}
    df = pd.DataFrame(data)
    
    return df.values if as_ndarray else df

def prepare_poly_data(*args, power):
    '''
    args: X, Xval, Xtest
    '''
    
    def prepare(x):
        df = poly_features(x, power)
        ndarr = normalize_feature(df).values
#         ndarr = ((df - df.mean()) / df.std()).values
        return np.insert(ndarr, 0, np.ones(ndarr.shape[0]), axis=1)
    
    return [prepare(x) for x in args]

X, y, Xval, yval, Xtest, ytest = load_data('HW5-Bias_and_Variance/ex5data1.mat')
X_p, Xval_p, Xtest_p = prepare_poly_data(X, Xval, Xtest, power=8)

def plot_learning_curve(X, y, Xval, yval, l=0):
    m = X.shape[0]
    training_cost, cv_cost = [], []
    
    for i in range(1, m+1):
        _x = X[:i, :]
        _y = y[:i]

        res = linear_regression(_x, _y, l=l)
        # 计算cost时不需要计算正则项，正则项只用于拟合
        tc = regularized_cost(res.x, _x, _y, l=0)
        cv = regularized_cost(res.x, Xval, yval, l=0)
        
        training_cost.append(tc)
        cv_cost.append(cv)
        
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(np.arange(1, m+1), training_cost, label='Training cost')
    ax.plot(np.arange(1, m+1), cv_cost, label='Cv cost')
    ax.legend(loc='best', title=r'$\lambda={}$'.format(l) )
    plt.show()

plot_learning_curve(X_p, y, Xval_p, yval, l=0)
plot_learning_curve(X_p, y, Xval_p, yval, l=1)
plot_learning_curve(X_p, y, Xval_p, yval, l=100)

candidate_l = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
training_cost, cv_cost, test_cost = [], [], []

for l in candidate_l:
    res = linear_regression(X_p, y, l)
    tc = regularized_cost(res.x, X_p, y, l=0)
    cv = regularized_cost(res.x, Xval_p, yval, l=0)
    test_c = regularized_cost(res.x, Xtest_p, ytest, l=0)
    
    training_cost.append(tc)
    cv_cost.append(cv)
    test_cost.append(test_c)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(candidate_l, training_cost, label='Training')
ax.plot(candidate_l, cv_cost, label='Cross validation')
ax.plot(candidate_l, test_cost, label='Testing')
ax.legend(loc='best')
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel('cost')
plt.show()

# 从验证集中找，最优的lambda=1
candidate_l[np.argmin(cv_cost)]
# 从测试集中找，最优的lambda=0.3
candidate_l[np.argmin(test_cost)]

