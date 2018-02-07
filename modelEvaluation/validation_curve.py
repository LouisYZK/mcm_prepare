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