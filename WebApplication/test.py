# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 21:58:58 2018

@author: Administrator
"""

clf = pickle.load(open(os.path.join('pkl_objects','classifier.pkl'), 'rb'))

import numpy as np
label = {0:'negative', 1:'positive'}
example = ['I love this movie']
X = vect.transform(example)
print('Prediction: %s\nProbability: %.2f%%' %(label[clf.predict(X)[0]],np.max(clf.predict_proba(X))*100))
