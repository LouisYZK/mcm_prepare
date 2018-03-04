# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 16:52:55 2018

@author: Administrator
"""

import re
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
text)
    text = (re.sub('[\W]+', ' ', text.lower()) +' '.join(emoticons).replace('-', ''))
    return text
#df['review'] = df['review'].apply(preprocessor)

def tokenizer(text):
    return text.split()
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

from nltk.corpus import stopwords
stop = stopwords.words('english')
