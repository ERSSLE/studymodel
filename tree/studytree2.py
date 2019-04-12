"""
决策树回归：
这里是一个一维数据的决策树回归，决策树试图拟合带有噪声点的正弦曲线。
因此，它通过局部回归来近似这条正弦曲线。
我们可以看到，树的最大深度越大，决策树越可以拟合（学习）数据的细节信息，
也即容易产生过拟合。
"""
# 导入所需要的模块及包
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

#生成随机数据
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80,1),axis=1)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

# 创建深度分别为2与5的两个回归树
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X,y)
regr_2.fit(X,y)
#模型训练
X_test = np.arange(0,5,0.01)[:,np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

#绘制图形
plt.scatter(X,y,s=20,edgecolor='black',c='darkorange',label='data')
plt.plot(X_test,y_1,color='cornflowerblue',label='max_depth=2',linewidth=2)
plt.plot(X_test,y_2,color='yellowgreen',label='max_depth=5',linewidth=2)
plt.xlabel('data')
plt.ylabel('target')
plt.title('Decision Tree Regression')
plt.legend()
plt.show()