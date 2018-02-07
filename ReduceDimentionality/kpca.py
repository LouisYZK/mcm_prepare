# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 16:56:37 2018

@author: Administrator
"""

from sklearn import datasets
X,y = datasets.make_circles(n_samples=1000,random_state=123, noise=0.1, factor=0.2)
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components=2,random_state=1,kernel='rbf',gamma=15)
X_pca = kpca.fit_transform(X)
import matplotlib.pyplot as plt
#fig,ax = plt.subplots(nrows=1,ncols=2)
#ax[0].scatter(X[y==0,0],X[y==0,1],color='red',marker='+')
#ax[0].scatter(X[y==1,0],X[y==1,1],color='blue',marker='^')
#ax[1].scatter(X_pca[y==0,0],X_pca[y==0,1],color='red',marker='+')
#ax[1].scatter(X_pca[y==1,0],X_pca[y==1,1],color='blue',marker='^')
#plt.show()
plt.scatter(X_pca[y==0,0],X_pca[y==0,1],color='red',marker='+')
plt.scatter(X_pca[y==1,0],X_pca[y==1,1],color='blue',marker='+')