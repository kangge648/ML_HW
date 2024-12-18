import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

np.random.seed(0)

# -------------------------data setting-------------------------
Samples = 100
Features = 50

True_theta = np.ones((Features + 1, 1))
True_theta[0] += 1
Theta_sum = sum(True_theta)

X = 50 * np.ones((Samples, Features)) + 10 * np.random.rand(Samples, Features)
new_X = np.ones((Samples, 1))
 
# 对特征进行标准化
scaler = StandardScaler()
X_scaled_features = scaler.fit_transform(X)
 
# 将标准化后的特征和截距项重新组合回 X
X = np.hstack((new_X, X_scaled_features))

Y = X @ True_theta

# X = (X - np.mean(X, axis = 0)) / np.std(X, axis = 0, ddof = 0)
# Y = (Y - np.mean(Y)) / np.std(Y)
# -------------------------data setting-------------------------

# -------------------------learning setting-------------------------
alpha = 0.01  # 学习率
num_iterations = 1000  # 迭代次数
 
Simu_Theta = np.zeros((Features + 1, 1))
 
losses = []
Theta_Losses = []
 
for i in range(num_iterations):
    Y_pred = X @ Simu_Theta
    
    loss = (1/2/Samples) * np.sum((Y_pred - Y) ** 2)
    losses.append(loss)
   
    gradient_0 = 1/Samples * np.sum(Y_pred - Y)  
    gradient_1 = 1/Samples * np.transpose(X_scaled_features) @ (Y_pred - Y)

    gradient = np.insert(gradient_1, 0, gradient_0)
    gradient = gradient.reshape(-1, 1) 
    Simu_Theta -= alpha * gradient

    Theta_Loss = Theta_sum - sum(Simu_Theta)
    Theta_Losses.append(Theta_Loss)
# -------------------------learning setting-------------------------

# -------------------------matrix setting-------------------------
matrix_theta = np.linalg.pinv(np.transpose(X) @ X) @ np.transpose(X) @ Y
print(matrix_theta)
# -------------------------matrix setting-------------------------


# -------------------------plot setting-------------------------
# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# 创建一个包含2行2列子图的图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 绘制损失函数变化曲线
ax1.plot(range(num_iterations), losses, color='blue')
ax1.set_title('损失函数随迭代次数的变化')
ax1.set_xlabel('迭代次数')
ax1.set_ylabel('损失值 (MSE)')
ax1.grid()

# 绘制参数总和 与 真实参数总和的误差
ax2.plot(range(num_iterations), Theta_Losses, color='blue')
ax2.set_title('参数误差随迭代次数的变化')
ax2.set_xlabel('迭代次数')
ax2.set_ylabel('差距值')
ax2.grid()

plt.tight_layout()  # 调整子图之间的间距
plt.show()
# -------------------------plot setting-------------------------