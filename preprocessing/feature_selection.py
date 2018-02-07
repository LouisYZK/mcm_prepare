# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 13:36:55 2018

@author: Administrator
"""

from sklearn import datasets
X = datasets.load_wine().data
y = datasets.load_wine().target
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,stratify = y)
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=500,random_state=1)
forest.fit(X_train,y_train)

from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(forest,threshold=0.1,prefit = True)
X_selected = sfm.transform(X_train)