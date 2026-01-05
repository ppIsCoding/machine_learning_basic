#使用随机梯度下降法来更新参数，与demo2不同
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

#均分误差
def MSE(x,y):
    return (1 / x.shape[0]) * np.sum((y - f(x)) ** 2)

#用随机值初始化参数
theta = np.random.rand(3)

#均分误差的历史记录
errors = []

#误差的差值
diff = 1

#学习率
ETA = 1e-3

#重复学习
errors.append(MSE(X, train_y))
while diff > 1e-2:
    #为了调整训练数据的顺序，准备随机的序列
    p = np.random.permutation(X.shape[0])
    #随机取出训练数据，使用随机梯度下降法更新参数
    for x, y in zip(X[p,:], train_y[p]):
        theta = theta - ETA * (f(x) - y) * x
    #计算误差
    errors.append(MSE(X, train_y))
    diff = errors[-2] - errors[-1]

#绘图
x = np.linspace(-3, 3, 100)
plt.plot(train_z, train_y, 'o')
plt.plot(x, f(to_matrix(x)))
plt.show()
