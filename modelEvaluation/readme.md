# Debugging Algorithms with Learning and Validation Curves
> 2.7晚上看了模型矫正模块的这两个曲线，获益匪浅，想记录一下

## 过、欠拟合问题
一个模型的过拟合和欠拟合与多种因素相关，首先过拟合的解决思路数据科学领域普遍认同的方法有：
- 增大训练集数量
- 加入超参数（hyperparameters）例如回归中的惩罚因子，来降低训练集的影响从而增大泛化能力
- 减少模型参数个数简化模型
- 降为维度，减少数据集噪音影响，可用feature selection 和feature extraction多种思路解决。

那么此次要探讨就是前两项对解决过拟合问题的效果的验证。

我们如何评判增加训练集是否对优化模型有用、以及如何选出最佳的超参数，都可以用到**sklearn的learning_curve** 和 **validation_curve**两种方法。

## Learning_curve
绘制出样本数量与 训练集准确率、验证集准确率的关系图。

将原训练集划分为数量依次增加的几个集合，每个集合采用K折交叉验证求出训练集的正确率和验证集的正确率。

```python
# learning_curver是一种诊断模型过拟合与训练集数量关系的对象，通过此手段可以知道针对当前模型增大训练集的样本数能否起到缓解过拟合的作用
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve ,train_test_split
from sklearn.pipeline import make_pipeline
import numpy as np 
# 下载乳腺癌数据集
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',header=None)
# 选取后面30个feature为数据集
X = df.iloc[:,2:].values
# 第2列为label
y = df.iloc[:,1].values
# label转化为数值
le = LabelEncoder()
y = le.fit_transform(y)
# 划分测试集、训练集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,stratify = y,random_state =1)
# 创建工作流管道 使用标准化、逻辑回归
pipe_lr = make_pipeline(StandardScaler(),LogisticRegression(C=1.0,random_state=1))
# 创建learning_curve对象
# train_sizes为训练样本书，从后面参数制定，这里按10%叠加10次
# trian_scores是训练集的准确度 cv=10代表交叉取10次
# test_scores是验证集(validation set)的准确度 同样交叉取10次
train_sizes, train_scores,test_scores = learning_curve(estimator = pipe_lr,X = X_train,y=y_train,train_sizes= np.linspace(0.1,1,10),cv =10,n_jobs=1)
# 10行代表10个样本规模的得分，各交叉验证了10次，所以下面按行取均值
train_mean = np.mean(train_scores,axis=1)
train_std = np.std(train_scores,axis=1)
test_mean = np.mean(test_scores,axis=1)
test_std = np.std(test_scores,axis=1)
# 画图，可视化样本数量与准确率之间的关系，同时也做出方差图
plt.plot(train_sizes,train_mean,color = 'blue',marker = 'o',markersize = 5,label = 'Training Accuracy')
plt.plot(train_sizes,test_mean,linestyle = '--',color = 'green',label = 'Validation Accuracy' )
plt.fill_between(train_sizes,train_mean+train_std,train_mean-train_std,color = 'blue',alpha = 0.5)
plt.fill_between(train_sizes,test_mean+test_std,test_mean-test_std,color = 'green',alpha =0.15)
plt.ylim(0.9,1)
plt.grid()
plt.legend()
plt.xlabel('Number of Training Sample')
plt.ylabel('Accuracy')
plt.show()
```
![](https://ws1.sinaimg.cn/large/6af92b9fgy1fo85s3y93rj20az07egm1.jpg)

在解释图形之前首先要说明：

我们期待一个泛化能力强的模型，在这里训练集的准确率高但验证集正确率低是一种过拟合的体现，而训练集准确率低又是一种潜拟合的体现，所以我们期望两个准确率尽可能地靠近

上图可以看出，在数量超过250之后，就可以看到两个准确率越来越接近98%左右，这证明了增大训练集个数是可以有效防止过拟合的。同时这个方法也告诉我们了模型的最佳训练集数量是多少。

## Validation curve
> 思想跟learning curve一样，不过此对象旨在探究最佳的超参数。超参数很常见的有回归中的正则化惩罚项、高斯核函数中的的gamma值等。

这次的例子是逻辑回归分类中的超参数惩罚因子C
$$J(w) = \sum_i^m{(y^(i) - f(w^Tx^(i))^2)} + C|w|^2$$

```python
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import validation_curve ,train_test_split
from sklearn.pipeline import make_pipeline
import numpy as np 
param_range = [0.001,0.01,0.1,1.0,10.0,100.0]
pipe_lr = make_pipeline(StandardScaler(),LogisticRegression(random_state=1))
train_scores,test_scores = validation_curve(estimator = pipe_lr,X= X_train,y=y_train,param_name = 'logisticregression__C',param_range = param_range,cv =10)
train_mean = np.mean(train_scores,axis=1)
train_std = np.std(train_scores,axis=1)
test_mean = np.mean(test_scores,axis=1)
test_std = np.std(test_scores,axis=1)
plt.plot(param_range,train_mean,color = 'blue',marker = 'o',markersize = 5,label = 'Training Accuracy')
plt.plot(param_range,test_mean,linestyle = '--',color = 'green',label = 'Validation Accuracy' )
plt.fill_between(param_range,train_mean+train_std,train_mean-train_std,color = 'blue',alpha = 0.15)
plt.fill_between(param_range,test_mean+test_std,test_mean-test_std,color = 'green',alpha =0.15)
plt.ylim(0.8,1.1)
plt.xscale('log')
plt.grid()
plt.legend()
plt.xlabel('Logistic Regression param C')
plt.ylabel('Accuracy')
plt.show()
```

![](https://ws1.sinaimg.cn/large/6af92b9fgy1fo864g4agsj20az07imxg.jpg)

要说明的是sklearn中的C的值是原公式中C的倒数，也就是图上的C越大惩罚力度越小。

图中可以看出，惩罚力度太大会造成准确率较低，惩罚因子需始终选择在0.1到1之间。