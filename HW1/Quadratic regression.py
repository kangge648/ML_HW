import numpy as np
import matplotlib.pyplot as plot

# 生成数据
np.random.seed(0)
 
# 生成自变量 X（房屋面积），范围从50到200平方米
X = 50 + 150 * np.random.rand(100)  # 生成从50到200的100个点
 
# 生成因变量 Y（房价），假设房价与房屋面积的关系
Y = 300000 + 2000 * X + np.random.randn(100) * 20000  # 线性关系加上噪声，价格范围在30万到50万之间
 

# 将数据标准化，帮助梯度下降更快收敛
X = (X - np.mean(X)) / np.std(X)
Y = (Y - np.mean(Y)) / np.std(Y)
 
# 梯度下降参数
alpha = 0.01  # 学习率
num_iterations = 1000  # 迭代次数
m = len(Y)  # 样本数量
 
# 初始化参数
theta_0 = 0  # 截距
theta_1 = 0  # 斜率
 
# 存储损失值
losses = []
 
# 梯度下降算法实现
for i in range(num_iterations):
    # 计算预测值
    Y_pred = theta_0 + theta_1 * X
    
    # 计算损失函数 (MSE)
    loss = (1/m) * np.sum((Y - Y_pred) ** 2)
    losses.append(loss)
    
    # 计算梯度
    gradient_0 = -(2/m) * np.sum(Y - Y_pred)  # 截距的梯度
    gradient_1 = -(2/m) * np.sum((Y - Y_pred) * X)  # 斜率的梯度
    
    # 更新参数
    theta_0 -= alpha * gradient_0
    theta_1 -= alpha * gradient_1
 
print(f'截距 (θ0): {theta_0:.4f}, 斜率 (θ1): {theta_1:.4f}')


# 设置matplotlib支持中文显示
plot.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plot.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# 创建一个包含2行2列子图的图形
fig, axs = plot.subplots(2, 2, figsize=(10, 8))
 
# 绘制生成的散点图
axs[0, 0].scatter(X, Y, color='blue', alpha=0.5)
axs[0, 0].set_title('房屋面积与房价的关系')
axs[0, 0].set_xlabel('房屋面积 (平方米)')
axs[0, 0].set_ylabel('房价 (人民币)')
axs[0, 0].grid()

# 绘制损失函数变化曲线
axs[0, 1].plot(range(num_iterations), losses, color='blue')
axs[0, 1].set_title('损失函数随迭代次数的变化')
axs[0, 1].set_xlabel('迭代次数')
axs[0, 1].set_ylabel('损失值 (MSE)')
axs[0, 1].grid()

# 可视化回归线
axs[1, 0].scatter(X, Y, color='blue', alpha=0.5)
axs[1, 0].plot(X, theta_0 + theta_1 * X, color='red', linewidth=2)
axs[1, 0].set_title('梯度下降后的线性回归拟合')
axs[1, 0].set_xlabel('房屋面积 (标准化)')
axs[1, 0].set_ylabel('房价 (标准化)')
axs[1, 0].grid()
 
plot.tight_layout()  # 调整子图之间的间距
plot.show()