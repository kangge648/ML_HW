import numpy as np

X = np.array([3, 1])
theta = np.array([[1, 4]])

Z = X @ theta.T
print(Z)
print(1/(1 + np.exp(Z)))