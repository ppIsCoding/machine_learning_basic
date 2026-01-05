import numpy as np
import matplotlib.pyplot as plt
#读入数据
train = np.loadtxt("click.csv",delimiter=',',skiprows=1)
train_x = train[:,0]
train_y = train[:,1]

#对数据进行预处理，标准化或者z-score规范化
mu = train_x.mean() #平均值
sigma = train_x.std() #标准差
def standardize(x):
    return (x - mu) / sigma

train_z = standardize(train_x)

#此次为多项式函数
#参数初始化
theta = np.random.rand(3)

#创建训练数据的矩阵
def to_matrix(x):
    return np.vstack([np.ones(x.shape[0]), x, x ** 2]).T

X = to_matrix(train_z)

#预测函数
def f(x):
    return np.dot(x, theta)
#目标函数
def E(x,y):
    return 0.5 * np.sum((y - f(x))**2)

#误差的差值
diff = 1
#学习率
ETA = 1e-3

#重复学习
error = E(X, train_y)
while diff > 1e-2:
    #更新参数
    theta = theta - ETA * np.dot(f(X) - train_y, X)
    #计算与上一次误差的差值
    current_error = E(X, train_y)
    diff = abs(error - current_error)
    error = current_error

#绘图
x = np.linspace(-3, 3, 100)
plt.plot(train_z, train_y, 'o')
plt.plot(x, f(to_matrix(x)))
plt.show()

#####################################################

#均分误差
def MSE(x,y):
    return (1 / x.shape[0]) * np.sum((y - f(x)) ** 2)

#用随机值初始化参数
theta = np.random.rand(3)

#均分误差的历史记录
errors = []

#误差的差值
diff = 1

#重复学习
errors.append(MSE(X, train_y))
while diff > 1e-2:
    # 更新参数
    theta = theta - ETA * np.dot(f(X) - train_y, X)
    # 计算与上一次误差的差值
    errors.append(MSE(X, train_y))
    diff = errors[-2] - errors[-1]

#绘制误差变化图
x = np.arange(len(errors))

plt.plot(x,errors)
plt.show()

