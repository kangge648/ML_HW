import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

theta = np.array([[1], [2], [3]])
means = np.array([[1], [1], [2]])
stds = np.array([[2], [2], [1]])
temp = means[:-1] * theta[1:] / stds[:-1]
theta[0] = (theta[0] - np.sum(temp)) * stds[-1] + means[-1]
theta[1:] = theta[1:] * stds[-1] / stds[:-1]
print(theta.shape, means.shape, stds.shape)