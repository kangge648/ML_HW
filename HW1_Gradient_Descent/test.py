import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Y = np.ones((2, 1))
Y_pred = 3 * np.ones((2,1))

X = np.matrix([[1, 8], [2, 6]])

print(np.sum(Y - Y_pred))
