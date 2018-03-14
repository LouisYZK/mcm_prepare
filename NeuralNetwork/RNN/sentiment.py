# preeprocessing the data
from collections import Counter
import pyprind
import pandas as pd 
from string import punctuation
import re 
import numpy as np 
counts = Counter()
df = pd.read_csv('movie_data.csv')
pbar = pyprind.ProgBar(len(df['review']))
for i,review in enumerate(df['review']):
    text = ''.join([c if c not in punctuation else ' '+c+' ' for c in review]).lower()
    df.loc[i,'review'] = text 
    pbar.update()
    counts.update(text.split())
# mapping each unique word into a integer
#counts是字典形式，词：数量
word_counts = sorted(counts,key = counts.get,reverse=True)
# 字典.get意思是取字典值进行排序
# 按照词频数排序后进行到整数的映射
word_to_int = {word:ii for ii,word in enumerate(word_counts,1)}
# enumerate(word_counts,1)是对字典从1开始枚举，也就是ii初始值为1
mapped_reviews = []
pbar = pyprind.ProgBar(len(df['review']))
for review in df['review']:
    mapped_reviews.append([word_to_int[word] for word in review.split()])
    pbar.update()
    

sequence_length = 200  ## sequence length (or T in our formulas)
sequences = np.zeros((len(mapped_reviews), sequence_length), dtype=int)
for i, row in enumerate(mapped_reviews):
    review_arr = np.array(row)
    sequences[i, -len(row):] = review_arr[-sequence_length:]

X_train = sequences[:25000, :]
y_train = df.loc[:25000, 'sentiment'].values
X_test = sequences[25000:, :]
y_test = df.loc[25000:, 'sentiment'].values

def bacth_generator(X,y=None,batch_size = 64):
    n_batch = len(X)//batch_size
    X = X[:n_batch*batch_size]
    if y is not None:
        y = y[:n_batch*batch_size]
    for ii in range(0,len(X),batch_size):
        if y is not None:
            yield X[ii:ii+batch_size],y[ii:ii+batch_size]
        else :
            yield X[ii:ii+batch_size]

import tensorflow as tf 
embedding = tf.Variable(tf.random_uniform(size=[n_unique,embedding_size],maxval=1,minval=-1))
embed_x = tf.nn.embedding_lookup(embedding,tf_x)