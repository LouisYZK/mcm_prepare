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