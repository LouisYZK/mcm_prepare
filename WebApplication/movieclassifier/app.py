from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import pymysql
import os
import numpy as np

# import HashingVectorizer from local dir
from vectorizer import vect

app = Flask(__name__)

######## Preparing the Classifier
cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir,
                 'pkl_objects',
                 'classifier.pkl'), 'rb'))
# db = os.path.join(cur_dir, 'reviews.sqlite')

def classify(document):
    label = {0: 'negative', 1: 'positive'}
    X = vect.transform([document])
    y = clf.predict(X)[0]
    proba = np.max(clf.predict_proba(X))
    return label[y], proba

def train(document, y):
    X = vect.transform([document])
    clf.partial_fit(X, [y])

def mysql_entry(document, y):
    conn = pymysql.connect(host = 'yzk.mysql.pythonanywhere-services.com',user = 'yzk',password = 'yangzhikai668',db = 'ML')
    c = conn.cursor()
    sql = 'insert into movie_text values(%s,%s)'
    c.execute(sql,(document,y))
    # c.execute("INSERT INTO review_db (review, sentiment, date)"\
    # " VALUES (?, ?, DATETIME('now'))", (document, y))
    conn.commit()
    conn.close()

######## Flask
class ReviewForm(Form):
    moviereview = TextAreaField('',
                                [validators.DataRequired(),
                                validators.length(min=15)])

@app.route('/')
def index():
    form = ReviewForm(request.form)
    return render_template('reviewform.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        # 得到评论 这个movieriew名称是怎么来的，下面我会在前端做下解释
        review = request.form['moviereview']
        y, proba = classify(review)
        # 在前端渲染结果
        return render_template('results.html',
                                content=review,
                                prediction=y,
                                probability=round(proba*100, 2))
    return render_template('reviewform.html', form=form)
# 反馈功能模块，用户点击是否情感判断正确，将用户给的正确标签和评论添加到数据库中，并使学习器学习
@app.route('/thanks', methods=['POST'])
def feedback():
    # 这些名称同样我在下面将前端时讲述
    feedback = request.form['feedback_button']
    review = request.form['review']
    prediction = request.form['prediction']

    inv_label = {'negative': 0, 'positive': 1}
    y = inv_label[prediction]
    if feedback == 'Incorrect':
        y = int(not(y))
    train(review, y)
    mysql_entry(review, y)
    return render_template('thanks.html')

if __name__ == '__main__':
    app.run(debug=True)