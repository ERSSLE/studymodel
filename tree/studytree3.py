"""
多输出决策树回归：
用一个例子来说明多输出决策树回归。
基于给出的一个属性，决策树被用于同时预测圆坐标的含有噪声的x值与y观测值。
因此，它通过局部回归来近似这条正弦曲线。
我们可以看到，树的最大深度越大，决策树越可以拟合（学习）数据的细节信息，
也即容易产生过拟合。
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

#
rng = np.random.RandomState(1)
X = np.sort(200 * rng.rand(100,1) - 100,axis=0)
y = np.array([np.pi * np.sin(X).ravel(),np.pi * np.cos(X).ravel()]).T
y[::5,:] += (0.5 - rnd.rand(20,2))

#
regr_1 = DecisionTreeRegressor(max_depth=2)