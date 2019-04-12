import numpy as np
import matplotlib.pyplot as plt

#设置字体黑体，汉文正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#导入鸢尾花数据函数，决策树模型
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 基本参数设定
n_classes = 3
plot_colors = "ryb"
plot_step = 0.02
# 加载数据
iris = load_iris()

#绘制决策树边界
for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
                                [1, 2], [1, 3], [2, 3]]):
    # 每次只选取两种属性
    X = iris.data[:, pair]
    y = iris.target
    # 训练
    clf = DecisionTreeClassifier().fit(X, y)
    # 绘制决策树边界
    plt.subplot(2, 3, pairidx + 1)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    #生成网格数据
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
    plt.xlabel(iris.feature_names[pair[0]])
    plt.ylabel(iris.feature_names[pair[1]])
    # 绘制原始数据点（按照数据真实分类）
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                    cmap=plt.cm.RdYlBu, edgecolor='black', s=15)
# 添加大标题用suptitle       
plt.suptitle("两两属性生长的决策树所形成的边界")
plt.legend(loc='best')
plt.axis('tight')

plt.figure()
#用所有数据所生成的决策树模型
clf = DecisionTreeClassifier().fit(iris.data,iris.target)
plot_tree(clf,filled=True)

#绘制混淆矩阵
from cm_plot import cm_plot
cm_plot(iris.target,clf.predict(iris.data))
plt.title('此决策树训练数据的再带入预测结果混淆矩阵')
plt.show()

