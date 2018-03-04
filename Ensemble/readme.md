# Using the Majority Voting Principle to make predictions
> 先介绍一种最简单的集成方法：多数投票，可以用概率知识证明多数投票的正确率要大于individual classifier；下文将用sklearn集成三种分类器并测试他们的分类正确率
## 数据准备
为作图方便，选取鸢尾花集的前两个feature
```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
iris = datasets.load_iris()
X, y = iris.data[50:, [1, 2]], iris.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.5,random_state = 1, stratify = y)
```
## individual classifier 训练与测试
这里选取三个分类器：逻辑回归、决策树、KNN;测试metric统一采用roc_auc,且用10折交叉验证
```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import numpy as np
clf1 = LogisticRegression(penalty='l2',C=0.001,random_state=1)
clf2 = DecisionTreeClassifier(max_depth=1,criterion='entropy',random_state=0)
clf3 = KNeighborsClassifier(n_neighbors=1,p=2,metric='minkowski')
# 将逻辑回归和KNN用pipe流封装，决策树不需要因为决策树不需要进行数据的标准化
pipe1 = Pipeline([['sc', StandardScaler()],['clf', clf1]])
pipe3 = Pipeline([['sc', StandardScaler()],['clf', clf3]])
clf_labels = ['Logistic regression', 'Decision tree', 'KNN']
print('10-fold cross validation:\n')
# 这里用的评判标准是roc_auc，当然也可以换做accuracy
for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
    scores = cross_val_score(estimator=clf, X=X_train,y=y_train,cv=10,scoring='roc_auc')
    print("ROC AUC: %0.2f (+/- %0.2f) [%s]"% (scores.mean(), scores.std(), label))
```
结果如下：

![](https://ws1.sinaimg.cn/large/6af92b9fgy1fp0os2hwn5j20e401uwef.jpg)

## Majority Voting集成并测试
多数集成可以直接调用sklearn.ensemble模块的VotingClassifier
```python
from sklearn.ensemble import VotingClassifier
vc = VotingClassifier(estimators=[('lr',pipe1),('dt',clf2),('knn',pipe3)],voting='soft')
# voting = 'soft'/'hard' soft是指用predict_prob_进行判断，hard严格按照多数个数来判断。因为下面的评判标准是roc_auc所以用soft
vc.fit(X_train,y_train)

clf_labels += ['Majority voting']
all_clf = [pipe1, clf2, pipe3, vc]
for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf,X=X_train,y=y_train,cv=10,scoring='roc_auc')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]"% (scores.mean(), scores.std(), label))
```
对比结果如下：

![](https://ws1.sinaimg.cn/large/6af92b9fgy1fp0ow3njl6j20e502qq2y.jpg)

可以看到Majority Voting Ensemble 分类器可以将auc提升至0.94

接着我们尝试绘制roc曲线：
```python
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
colors = ['black', 'orange', 'blue', 'green']
linestyles = [':', '--', '-.', '-']
for clf, label, clr, ls in zip(all_clf, clf_labels, colors, linestyles):
    # assuming the label of the positive class is 1
    y_pred = clf.fit(X_train,y_train).predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true=y_test,y_score=y_pred)
    roc_auc = auc(x=fpr, y=tpr)
    plt.plot(fpr, tpr,color=clr,linestyle=ls,label='%s (auc = %0.2f)' % (label, roc_auc))
plt.legend(loc='lower right')
plt.plot([0, 1], [0,1],linestyle='--',color='gray',linewidth=2)
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.grid(alpha=0.5)
plt.xlabel('False positive rate (FPR)')
plt.ylabel('True positive rate (TPR)')
plt.show()
```
如下：
![](https://ws1.sinaimg.cn/large/6af92b9fgy1fp0p1zyq5cj20at07egm2.jpg)

到这里似乎可以结束了，但是0.94的auc能否再提高呢？可以看到前面三个individual classifier在训练时的参数是我自己给定的，那么就可以用chapter6中学到的Hyperparameters Tunningd的方法找到最佳参数
## Majority Voting Ensemble Hyperparams Tunning
首先调用VotingClassifer的get_param方法查看集成的分类器的参数概况：
```python
{'dt': DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=1,
             max_features=None, max_leaf_nodes=None,
             min_impurity_decrease=0.0, min_impurity_split=None,
             min_samples_leaf=1, min_samples_split=2,
             min_weight_fraction_leaf=0.0, presort=False, random_state=0,
             splitter='best'),
 'dt__class_weight': None,
 'dt__criterion': 'entropy',
 'dt__max_depth': 1,
 'dt__max_features': None,
 'dt__max_leaf_nodes': None,
 'dt__min_impurity_decrease': 0.0,
 'dt__min_impurity_split': None,
 'dt__min_samples_leaf': 1,
 'dt__min_samples_split': 2,
 'dt__min_weight_fraction_leaf': 0.0,
 'dt__presort': False,
 'dt__random_state': 0,
 'dt__splitter': 'best',
 'estimators': [('lr', Pipeline(memory=None,
        steps=[('sc', StandardScaler(copy=True, with_mean=True, with_std=True)), ['clf', LogisticRegression(C=0.001, class_weight=None, dual=False, fit_intercept=True,
             intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
             penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
             verbose=0, warm_start=False)]])),
  ('dt',
   DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=1,
               max_features=None, max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, presort=False, random_state=0,
               splitter='best')),
  ('knn', Pipeline(memory=None,
        steps=[('sc', StandardScaler(copy=True, with_mean=True, with_std=True)), ['clf', KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
              metric_params=None, n_jobs=1, n_neighbors=1, p=2,
              weights='uniform')]]))],
 'flatten_transform': None,
 'knn': Pipeline(memory=None,
      steps=[('sc', StandardScaler(copy=True, with_mean=True, with_std=True)), ['clf', KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
            metric_params=None, n_jobs=1, n_neighbors=1, p=2,
            weights='uniform')]]),
 'knn__clf': KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
            metric_params=None, n_jobs=1, n_neighbors=1, p=2,
            weights='uniform'),
 'knn__clf__algorithm': 'auto',
 'knn__clf__leaf_size': 30,
 'knn__clf__metric': 'minkowski',
 'knn__clf__metric_params': None,
 'knn__clf__n_jobs': 1,
 'knn__clf__n_neighbors': 1,
 'knn__clf__p': 2,
 'knn__clf__weights': 'uniform',
 'knn__memory': None,
 'knn__sc': StandardScaler(copy=True, with_mean=True, with_std=True),
 'knn__sc__copy': True,
 'knn__sc__with_mean': True,
 'knn__sc__with_std': True,
 'knn__steps': [('sc',
   StandardScaler(copy=True, with_mean=True, with_std=True)),
  ['clf',
   KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
              metric_params=None, n_jobs=1, n_neighbors=1, p=2,
              weights='uniform')]],
 'lr': Pipeline(memory=None,
      steps=[('sc', StandardScaler(copy=True, with_mean=True, with_std=True)), ['clf', LogisticRegression(C=0.001, class_weight=None, dual=False, fit_intercept=True,
           intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
           penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
           verbose=0, warm_start=False)]]),
 'lr__clf': LogisticRegression(C=0.001, class_weight=None, dual=False, fit_intercept=True,
           intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
           penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
           verbose=0, warm_start=False),
 'lr__clf__C': 0.001,
 'lr__clf__class_weight': None,
 'lr__clf__dual': False,
 'lr__clf__fit_intercept': True,
 'lr__clf__intercept_scaling': 1,
 'lr__clf__max_iter': 100,
 'lr__clf__multi_class': 'ovr',
 'lr__clf__n_jobs': 1,
 'lr__clf__penalty': 'l2',
 'lr__clf__random_state': 1,
 'lr__clf__solver': 'liblinear',
 'lr__clf__tol': 0.0001,
 'lr__clf__verbose': 0,
 'lr__clf__warm_start': False,
 'lr__memory': None,
 'lr__sc': StandardScaler(copy=True, with_mean=True, with_std=True),
 'lr__sc__copy': True,
 'lr__sc__with_mean': True,
 'lr__sc__with_std': True,
 'lr__steps': [('sc',
   StandardScaler(copy=True, with_mean=True, with_std=True)),
  ['clf',
   LogisticRegression(C=0.001, class_weight=None, dual=False, fit_intercept=True,
             intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
             penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
             verbose=0, warm_start=False)]],
 'n_jobs': 1,
 'voting': 'soft',
 'weights': None}
```
查看这个有两个原因：
- 得知个体分类器参数的初始化值
- 得知超参数的表达 如逻辑回归的惩罚因子的表述是'lr__clf__C'，得知这个是为了下面的采用GridSearch配置参数名 （PS:之所以用双下划线我认为是参数名有很多带有下划线的，所以为了避免冲突sklearn统一采用了双下划线）

接下来就要用前面提到的GridSearch方法找到最佳参数：
```python
from sklearn.model_selection import GridSearchCV
# 这里测试两个参数：决策树的深度和逻辑回归的惩罚因子
params = {'dt__max_depth': [1, 2],'lr__clf__C': [0.001, 0.1, 100.0]}
grid = GridSearchCV(estimator=vc,param_grid=params,cv=10,scoring='roc_auc')
grid.fit(X_train, y_train)
for params, mean_score, scores in grid.grid_scores_:
    print("%0.3f+/-%0.2f %r"% (mean_score, scores.std() / 2, params))
print('Accuracy: %.2f' % grid.best_score_)
```
不出意外应该有得出六种情况的结果：

![](https://ws1.sinaimg.cn/large/6af92b9fgy1fp0pgg1jntj20fi03ujrm.jpg)

可以看到最佳的参数是一层决策树、C= 100（**这里sklearnd的C与原公式中的C是倒数关系，也就是说这里的100代表一个很小的惩罚力度**）。在这里能将auc从0.94提升到0.97,调整超参数还是有很大效果的。
