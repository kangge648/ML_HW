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

# one-hot 编码
def expend_y(y):
    res = []
    for i in y:
        tmp = np.zeros(10)
        tmp[i-1] = 1
        res.append(tmp)
    return np.array(res)

def load_weight(path):
    data = loadmat(path)
    return data['Theta1'], data['Theta2']

def serialize(a, b):
    return np.concatenate((np.ravel(a), np.ravel(b)))

def deserialize(seq):
    return seq[ : 25*401].reshape(25, 401), seq[25*401 : ].reshape(10, 26)

raw_x, raw_y = load_data('HW4-NN_back_propagation/ex4data1.mat')
# plot_100_image(raw_x)
X, y = load_data('HW4-NN_back_propagation/ex4data1.mat', transpose=False)
X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
y = expend_y(y)

t1, t2 = load_weight('HW4-NN_back_propagation/ex4weights.mat')
theta = serialize(t1, t2)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def feed_forward(theta, X):
    t1, t2 = deserialize(theta) # t1:(25, 401) t2:(10, 26)
    a1 = X # 5000*401
    
    z2 = a1 @ t1.T # 5000*25
    a2 = np.insert(sigmoid(z2), 0, np.ones(z2.shape[0]), axis=1) # 5000*26
    
    z3 = a2 @ t2.T # 5000*10
    h = sigmoid(z3) # 5000*10
    return a1, z2, a2, z3, h

h = feed_forward(theta, X)[-1]

def cost(theta, X, y):
    h = feed_forward(theta, X)[-1]
    tmp = -y * np.log(h) - (1-y) * np.log(1-h)
    return tmp.sum() / y.shape[0]

def regularized_cost(theta, X, y, l=1):
    t1, t2 = deserialize(theta)
    m = X.shape[0]
    
    reg1 = np.power(t1[:, 1:], 2).sum() / (2 * m)
    reg2 = np.power(t2[:, 1:], 2).sum() / (2 * m)
    
    return cost(theta, X, y) + reg1 + reg2

def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

def gradient(theta, X, y):
    t1, t2 = deserialize(theta)
    m = X.shape[0]
    
    delta1 = np.zeros(t1.shape) # 25*401
    delta2 = np.zeros(t2.shape) # 10*26
    
    a1, z2, a2, z3, h = feed_forward(theta, X)
    
    for i in range(m):
        a1i = a1[i] # 1*401
        z2i = z2[i] # 1*25
        a2i = a2[i] # 1*26

        hi  = h[i]  # 1*10
        yi  = y[i]  # 1*10
        d3i = hi - yi # 1*10，输出层的误差
        
        z2i = np.insert(z2i, 0, np.ones(1))
        d2i = t2.T @ d3i * sigmoid_gradient(z2i) # 1*26 隐藏层的误差
        
        # careful with np vector transpose
        delta2 += np.matrix(d3i).T @ np.matrix(a2i)
        delta1 += np.matrix(d2i[1:]).T @ np.matrix(a1i)
    
    return serialize(delta1, delta2)

d1, d2 = deserialize(gradient(theta, X, y))

def regularized_gradient(theta, X, y, l=1):
    m = X.shape[0]
    delta1, delta2 = deserialize(gradient(theta, X, y))
    delta1 /= m
    delta2 /= m
    
    t1, t2 = deserialize(theta)
    t1[:, 0] = 0
    t2[:, 0] = 0
    
    delta1 += l / m * t1
    delta2 += l / m * t2
    
    return serialize(delta1, delta2)

def expand_array(arr):
    '''
    replicate array into matrix
    
    [1, 2, 3]
    
    [[1, 2, 3],
    [1, 2, 3],
    [1, 2, 3]]
    
    '''
    
    return np.array(np.matrix(np.ones(arr.shape[0])).T @ np.matrix(arr))

def gradient_checking(theta, X, y, epsilon, regularized=False):
    m = len(theta)
    def a_numeric_grad(plus, minus, regularized=False):
        if regularized:
            return (regularized_cost(plus, X, y) - regularized_cost(minus, X, y)) / (epsilon*2)
        else:
            return (cost(plus, X, y) - cost(minus, X, y)) / (epsilon*2)
    
    theta_matrix = expand_array(theta)
    epsilon_matrix = np.identity(m) * epsilon # identity单位矩阵
    plus_matrix = theta_matrix + epsilon_matrix
    minus_matrix = theta_matrix - epsilon_matrix
    
    approx_grad = np.array([a_numeric_grad(plus_matrix[i], minus_matrix[i], regularized) 
                            for i in range(m)])
    analytic_grad = regularized_gradient(theta, X, y) if regularized else gradient(theta, X, y)
    diff = np.linalg.norm(approx_grad - analytic_grad) / np.linalg.norm(approx_grad + analytic_grad)
    
    print('If your backpropagation implementation is correct,\nthe relative difference will be smaller than 10e-9 (assume epsilon=0.0001).\nRelative Difference: {}\n'.format(diff))

gradient_checking(theta, X, y, epsilon=0.0001, regularized=True) # 慢

def random_init(size):
    return np.random.uniform(-0.12, 0.12, size)

def nn_training(theta, X, y):
    init_theta = random_init(len(theta))
    res = opt.minimize(fun=regularized_cost, x0 = init_theta,
                      args=(X, y, 1), method='TNC',
                      jac=regularized_gradient,
                      options={'maxiter': 400})
    return res

res = nn_training(theta, X, y) # 慢
final_theta = res.x
_, _, _, _, h = feed_forward(theta, X)
y_pred = np.argmax(h, axis=1) + 1
print(classification_report(raw_y, y_pred))

def plot_hidden_layer(theta):
    t1, t2 = deserialize(theta)
    hidden_layer = t1[:, 1:]
    fig, ax_array = plt.subplots(nrows=5, ncols=5, sharey=True, sharex=True, figsize=(5,5))
    for r in range(5):
        for c in range(5):
            ax_array[r, c].matshow(hidden_layer[5*r+c].reshape((20,20)), 
                                  cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))            
    plt.show()

plot_hidden_layer(final_theta)