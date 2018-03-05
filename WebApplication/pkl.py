# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 21:47:10 2018

@author: Administrator
"""
import pickle
import os
dest = os.path.join('movieclassifier', 'pkl_objects')
if not os.path.exists(dest):
    os.makedirs(dest)
pickle.dump(stop,open(os.path.join(dest, 'stopwords.pkl'),'wb'),protocol=4)
pickle.dump(clf,open(os.path.join(dest, 'classifier.pkl'), 'wb'),protocol=4)