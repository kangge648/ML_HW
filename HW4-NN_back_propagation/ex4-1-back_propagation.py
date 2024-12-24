import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn.metrics import classification_report


def load_data(path, transpose=True):
    data = loadmat(path)
    X = data['X']
    y = data['y']
    y = y.reshape(y.shape[0])
    
    if transpose:
        X = np.array([im.reshape((20,20)).T.reshape(400) for im in X])
    return X, y

def plot_100_image(X):
    sz = int(np.sqrt(X.shape[1]))
    sample_idx = np.random.choice(np.arange(X.shape[0]), 100)
    sample_images = X[sample_idx, :]
    
    fig, axs = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, 
                            figsize=(8,8))
    for r in range(10):
        for c in range(10):
            axs[r, c].matshow(sample_images[10 * r + c].reshape((sz, sz)),
                              cmap=matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show()    

raw_x, raw_y = load_data('HW4-NN_back_propagation/ex4data1.mat')
# plot_100_image(raw_x)
X, y = load_data('HW4-NN_back_propagation/ex4data1.mat', transpose=False)
X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
