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

#创建随机数据集
rng = np.random.RandomState(1)
X = np.sort(200 * rng.rand(100,1) - 100,axis=0)
y = np.array([np.pi * np.sin(X).ravel(),np.pi * np.cos(X).ravel()]).T
y[::5,:] += (0.5 - rng.rand(20,2))

#创建回归模型
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_3 = DecisionTreeRegressor(max_depth=8)
regr_1.fit(X,y)
regr_2.fit(X,y)
regr_3.fit(X,y)

#不同深度决策树预测
X_test = np.arange(-100.0,100.0,0.01)[:,np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)
y_3 = regr_3.predict(X_test)

#预测结果可视化
plt.figure()
s=25
plt.scatter(y[:,0],y[:,1],c='navy',s=s,edgecolor='black',
            label='data')
plt.scatter(y_1[:,0],y_1[:,1],c='cornflowerblue',s=s,
            edgecolor='black',label='max_depth=2')
plt.scatter(y_2[:,0],y_2[:,1],c='red',s=s,
            edgecolor='black',label='max_depth=5')
plt.scatter(y_3[:,0],y_3[:,1],c='orange',s=s,
            edgecolor='black',label='max_depth=8')
plt.xlim([-6,6])
plt.ylim([-6,6])
plt.xlabel('target1')
plt.ylabel('target2')
plt.title('Muti-output Decison Tree Regression')
plt.legend(loc='best')
plt.show()






