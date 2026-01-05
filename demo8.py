# 正则化

import numpy as np
import matplotlib.pyplot as plt

# 真正的函数 - 这是我们试图逼近的目标函数
# 这里使用了一个三次多项式函数作为真实模型
def g(x):
    return 0.1 * (x ** 3 + x ** 2 + x)

# 生成训练数据
# train_x: 在[-2, 2]区间内均匀分布的8个点
# train_y: 在真实函数基础上添加了少量噪声的训练标签
train_x = np.linspace(-2, 2, 8)
train_y = g(train_x) + np.random.randn(train_x.size) * 0.05

# 用于绘图的x轴数据，更密集以获得平滑曲线
x = np.linspace(-2, 2, 100)
# plt.plot(train_x, train_y, 'o')
# plt.plot(x, g(x), linestyle='dashed')
# plt.ylim(-1, 2)
# plt.show()

##########不使用正则化，用多项式回归，让结果出现overfitting

# 数据标准化 - 将数据转换为均值为0、标准差为1的分布
# 这有助于提高梯度下降的收敛速度和稳定性
mu = train_x.mean()
sigma = train_x.std()
def standardize(x):
    return (x - mu) / sigma

# 对训练数据进行标准化
train_z = standardize(train_x)

# 将输入数据转换为高次多项式矩阵
# 创建一个10次多项式回归模型（包含偏置项）
# 这样可以拟合非常复杂的函数，但也容易过拟合
def to_matrix(x):
    return np.vstack([
    np.ones(x.size),  # 偏置项
    x,                # 一次项
    x ** 2,           # 二次项
    x ** 3,           # 三次项
    x ** 4,           # 四次项
    x ** 5,           # 五次项
    x ** 6,           # 六次项
    x ** 7,           # 七次项
    x ** 8,           # 八次项
    x ** 9,           # 九次项
    x ** 10,          # 十次项
]).T

# 构建训练数据矩阵
X = to_matrix(train_z)

# 初始化模型参数（权重）
theta = np.random.randn(X.shape[1])
# shape可以取到矩阵的行数和列数

# 预测函数 - 使用线性模型进行预测
def f(x):
    return np.dot(x, theta)
# dot函数是点乘，矩阵乘法

# 目标函数（损失函数）- 均方误差
# 衡量模型预测值与真实值之间的差距
def E(x, y):
    return 0.5 * np.sum((y - f(x)) ** 2)

# 学习率 - 控制梯度下降的步长
ETA = 1e-4

# 误差变化量 - 用于判断是否停止训练
diff = 1

# 重复学习（梯度下降）
# 持续更新参数直到误差变化小于阈值
error = E(X, train_y)
while diff > 1e-6:
    # 计算梯度并更新参数
    # 梯度为 (预测值 - 真实值) * 特征矩阵
    theta = theta - ETA * np.dot(f(X) - train_y, X)
    current_error = E(X, train_y)
    diff = error - current_error
    error = current_error

# 对结果绘图 - 绘制无正则化模型的拟合效果
z = standardize(x)
plt.plot(train_z, train_y, 'o')
plt.plot(z, f(to_matrix(z)))
plt.show()


# 保存未正则化的参数，然后再次参数初始化
theta1 = theta
theta = np.random.randn(X.shape[1])

# 正则化常量 - 控制正则化项的强度
LAMBDA = 1

# 误差变化量初始化
diff = 1

# 重复学习（包含正则化项）
# 使用L2正则化（Ridge回归）来防止过拟合
error = E(X, train_y)
while diff > 1e-6:
    # 正则化项。偏置项不适用正则化，所以为 0
    # L2正则化：对除偏置项外的所有权重进行平方和惩罚
    reg_term = LAMBDA * np.hstack([0, theta[1:]])
    # 应用正则化项，更新参数
    # 梯度 = 原始梯度 + 正则化项
    theta = theta - ETA * (np.dot(f(X) - train_y, X) + reg_term)
    current_error = E(X, train_y)
    diff = error - current_error
    error = current_error

# 对结果绘图 - 绘制带正则化模型的拟合效果
plt.plot(train_z, train_y, 'o')
plt.plot(z, f(to_matrix(z)))
plt.show()



theta2 = theta

plt.plot(train_z, train_y, 'o')

# 画出未应用正则化的结果
theta = theta1
plt.plot(z, f(to_matrix(z)), linestyle='dashed')

# 画出应用了正则化的结果
theta = theta2
plt.plot(z, f(to_matrix(z)))

plt.show()