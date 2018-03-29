# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 13:09:29 2018

@author: Administ
"""
from sklearn import datasets
X = datasets.load_iris().data[:,0:2]
y = datasets.load_iris().target
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,stratify = y)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
from sklearn.svm import SVC
svm = SVC(kernel='rbf',C=2.0,random_state=1,gamma=1)
svm.fit(X_train_std,y_train)
#gamma越小越边界soft