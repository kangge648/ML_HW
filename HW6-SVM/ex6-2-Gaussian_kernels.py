import numpy as np
import pandas as pd
import sklearn.svm
import seaborn as sns
import scipy.io as sio
import matplotlib.pyplot as plt

def gaussian_kernel(x1, x2, sigma):
    return np.exp(- np.power(x1-x2, 2).sum() / (2*sigma**2))


x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2

gaussian_kernel(x1, x2, sigma)

mat = sio.loadmat('HW6-SVM/data/ex6data2.mat')
print(mat.keys())

data = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
data['y'] = mat.get('y')
data.head()

positive = data[data.y == 1]
negative = data[data['y'] == 0]

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['X1'], positive['X2'], label='positive', s=50, marker='+', c='r')
ax.scatter(negative['X1'], negative['X2'], label='negative', s=50, marker='o', c='b')
ax.legend(loc='best')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
plt.show()

svc1 = sklearn.svm.SVC(C=100, kernel='rbf', gamma=1, probability=True)
svc1.fit(data[['X1', 'X2']], data['y'])
svc1.score(data[['X1', 'X2']], data['y'])
positive = data[data.y == 1]
negative = data[data['y'] == 0]

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['X1'], positive['X2'], label='positive', s=50, marker='+', c='r')
ax.scatter(negative['X1'], negative['X2'], label='negative', s=50, marker='o', c='b')
ax.legend(loc='best')
ax.set_xlabel('X1')
ax.set_ylabel('X2')

# 决策边界, 使用等高线表示
x1 = np.arange(0.04, 1, 0.01)
x2 = np.arange(0.4, 1, 0.01)
x1, x2 = np.meshgrid(x1, x2)
y_pred = np.array([svc1.predict(np.vstack((a, b)).T) for (a, b) in zip(x1, x2)])
plt.contour(x1, x2, y_pred, colors='g', linewidths=.5)
plt.show()

# predict_proba = svc.predict_proba(data[['X1', 'X2']])[:, 0]
predict_proba = svc1.predict_proba(data[['X1', 'X2']])[:, 1]

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(data['X1'], data['X2'], s=50, c=predict_proba, cmap='Reds')
ax.set_xlabel('X1')
ax.set_ylabel('X2')

plt.show()

svc10 = sklearn.svm.SVC(C=100, kernel='rbf', gamma=10, probability=True)
svc10.fit(data[['X1', 'X2']], data['y'])
svc10.score(data[['X1', 'X2']], data['y'])
positive = data[data.y == 1]
negative = data[data['y'] == 0]

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['X1'], positive['X2'], label='positive', s=50, marker='+', c='r')
ax.scatter(negative['X1'], negative['X2'], label='negative', s=50, marker='o', c='b')
ax.legend(loc='best')
ax.set_xlabel('X1')
ax.set_ylabel('X2')

# 决策边界, 使用等高线表示
x1 = np.arange(0.04, 1, 0.01)
x2 = np.arange(0.4, 1, 0.01)
x1, x2 = np.meshgrid(x1, x2)
y_pred = np.array([svc10.predict(np.vstack((a, b)).T) for (a, b) in zip(x1, x2)])
plt.contour(x1, x2, y_pred, colors='g', linewidths=.5)
plt.show()


# predict_proba = svc.predict_proba(data[['X1', 'X2']])[:, 0]
predict_proba = svc10.predict_proba(data[['X1', 'X2']])[:, 1]

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(data['X1'], data['X2'], s=50, c=predict_proba, cmap='Reds')
ax.set_xlabel('X1')
ax.set_ylabel('X2')

plt.show()





















