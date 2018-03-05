# Ch9:Embedding a Machine Learning Model into a Web Application
> 前面的ML模型都是在本地运行计算的，此chapter介绍如何将模型应用在Web app上以获得实时学习、应用
**这一章主要偏机器学习功能Web的简单开发，与算法关系不大。没有跳过反而收获不少。不得不佩服作者在构思’算法+系统‘时所作出的为读者充分考虑的思想**

本章主要实现下列功能：（将情感分析学习器上线Web，使之能即时搜集新数据、学习新数据、更新学习器、做出预测。最后作者还提供了一种上传到公共Web空间的一种简便方式）

- Saving the current state of a trained machine learning model
- Using databases for data storage 
- Developing a web application using the popular **Flask** web framework
- Deploying a machine learning application to a public web server

## Serializing fitted scikit-learn estimators
> 这里应用的是 sentiment analysis 中的out_of_core.py中训练的线上模型为例，关于这个学习模型主要是一个one_line的学习模式，就是所有数据不在本地内存中。

### 即时学习的方法
fit 是全部学习所有的数据，而estimator的part_fit(X_new)方法可以在原来的学习器基础上学习新数据，更新新性能。
### 序列化
序列化将本地变量（如estimator）存储为二进制pkl文件，当大型的estimator需要在网络空间使用时直接dump出pkl,这一过程的速度回很快。（想想上章节中本地memory学习器学习完需要40min,而on_line训练完成也需要3min,这放在Web服务器上是不可能的，所以序列化是必要的）
```python
import pickle
import os
dest = os.path.join('movieclassifier', 'pkl_objects')
if not os.path.exists(dest):
    os.makedirs(dest)
# 为了方便，将停用词也序列化
pickle.dump(stop,open(os.path.join(dest, 'stopwords.pkl'),'wb'),protocol=4)
pickle.dump(clf,open(os.path.join(dest, 'classifier.pkl'), 'wb'),protocol=4)
```
## 数据库
作者用的是Sqlite，emmmmmm一言难尽啊

**Sqlite** ，对就是Django初始化使用的轻量级数据库，不需要额外的配置接口什么的，可以像访问一个文件一样地访问他。。。 多么厉害友好！

所以，我决定换掉它，用Mysql

至于为啥，习惯吧......  作者的初衷是好的，因为这样免去了学习新数据库的成本和连接耽误的风险，作者面向的对象是没有Web开发经验的人，所以这一点是值得肯定的。

## Flask框架
Why Flask？？作者是这样回答的：
> Flask is also known as a microframework, which means that its core is kept lean
and simple but can be easily extended with other libraries. Although the learning
curve of the lightweight Flask API is not nearly as steep as those of other popular
Python web frameworks, such as Django, I encourage you to take a look at the
official Flask documentation at http://flask.pocoo.org/docs/0.12/ to learn more about
its functionality.

哦，你说Flask比Django简单，好吧那就用Flask吧！

### 控制层
本质上讲 Flask也是用的MVC理论。

Flask最主要的就是app.py文件 相当于Django的views.py ,不同的是他还负责路径和数据库（Flask数据库靠依赖三方库，所以还需要开发者自行编写SQL语句）的问题。而在Django中路径需要有专门的配置文件，数据库可以用ORM。

这么一对比Django的好处是数据库可以用映射关系避免写sql,模块功能更加清楚分明。

但Flask也一定有很多比Django好很多的地方，刚接触Flask希望在未来慢慢体验吧

app.py代码，重要地方我做了注释：
```python
from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import pymysql
import os
import numpy as np

# import HashingVectorizer from local dir
from vectorizer import vect

app = Flask(__name__)

######## Preparing the Classifier 必要的函数准备
cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir,
                 'pkl_objects',
                 'classifier.pkl'), 'rb'))
# db = os.path.join(cur_dir, 'reviews.sqlite')

def classify(document):
    label = {0: 'negative', 1: 'positive'}
    # 转化词袋模型
    X = vect.transform([document])
    # 得到预测结果
    y = clf.predict(X)[0]
    # 得到预测结果的准确率
    proba = np.max(clf.predict_proba(X))
    return label[y], proba

def train(document, y):
    X = vect.transform([document])
    # 即时学习新的数据
    clf.partial_fit(X, [y])

def mysql_entry(document, y):
    # 这个模块不解释，添加新影评进入数据库
    conn = pymysql.connect(host = 'yzk.mysql.pythonanywhere-services.com',user = 'yzk',password = '',db = 'ML')
    c = conn.cursor()
    sql = 'insert into movie_text values(%s,%s)'
    c.execute(sql,(document,y))
    # c.execute("INSERT INTO review_db (review, sentiment, date)"\
    # " VALUES (?, ?, DATETIME('now'))", (document, y))
    conn.commit()
    conn.close()

######## 从这里Flask部分开始，下面会调用上面的函数
class ReviewForm(Form):
    moviereview = TextAreaField('',
                                [validators.DataRequired(),
                                validators.length(min=15)])
# Flask用装饰器控制路径和方法
# 用户输入评论模块
 @app.route('/')
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
```
### 前端
Flask的前端跟Django的理念差不多，渲染模板。这里讲一下上段代码中Form 在前端的表现问题：

关于输入文本框没有那么简单，还需要很多额外的功能，如限定最大最小字符数，错误返回，样式等等一系列问题，这些再去编写相关css\js麻烦，Flask和Django都会用到一些如Jinja的成熟模板：
```html
<!-- _formhelpers.html是引入Jinja -->
{% from "_formhelpers.html" import render_field %}

<form method=post action="/results">
  <dl>
     <!-- 这里传入了样式参数 ，表单名称就是从这里来的-->
	{{ render_field(form.moviereview, cols='30', rows='10') }}
  </dl>
  <div>
	  <input type=submit value='Submit review' name='submit_btn'>
  </div>
</form>
```
前端与后端 参数传递 也跟Django的模式差不多。

### 在本地运行起来吧！
输入影评

![](https://ws1.sinaimg.cn/large/6af92b9fgy1fp2c97k713j20ap0cpdg4.jpg)

得到结果

![](https://ws1.sinaimg.cn/large/6af92b9fgy1fp2cahj5kqj20pk0bmmy6.jpg)

可以看到得出了高概率的消极结果,....What??? 这里先对模型准确性不做判述；用户点击正确或者不正确可以添加新数据到数据库中，然后学习器及时学习他。

循环测试了四次，发现此分类器对于短一些的情感色彩不那么明显的评论是无能为力的

打开本地Mysql后台，测试一下是否添加新数据：

![](https://ws1.sinaimg.cn/large/6af92b9fgy1fp2cdz54lgj20l6063jrg.jpg)

嗯成功添加了。

## 上传到公共空间吧！
> 其实作者写道这里是画蛇添足了，没必要继续下去了，本地运行成功的学习成本已经很大了。不过他就写了两页，我看完之后打开了新世界。。。

作者推荐了**pythonAnywhere**平台，跟google\amozon之类的云平台不同，他主要面对python的Web服务，在这里你可以免费\付费 地部署自己的Python - Web 应用，Flask,Django都可以。因为专门针对Python，所以对于开发者来说极大的便利就是：**不需要再搭建环境！**，只需要吧我们上面本地做好的文件上传就可以访问我们的机器学习应用啦！

[点这里访问上线的情感分析程序](http://yzk.pythonanywhere.com)
就像这样：

![](https://ws1.sinaimg.cn/large/6af92b9fgy1fp2cjzipfdj20wi0bsjsx.jpg)

然后再Web中创建自己的应用：

![](https://ws1.sinaimg.cn/large/6af92b9fgy1fp2cl50bruj20g60h1q52.jpg)

简直不能再方便了，上图中可以看到：通用网关、日志文件、静态文件路径都给你放好了...... 真的省了一堆麻烦.

等等！，数据库咋办？ 总不能局域网访问吧？？（看到这里对作者为何使用Sqlite恍然大悟）

这个问题pythonAnywhere当然想到了，你可以在他们云端免费申请一个Mysql或者Postgres数据库。

![](https://ws1.sinaimg.cn/large/6af92b9fgy1fp2cohz0hcj20gg0eggmt.jpg)

上图可知，简直不需要你再做什么了，唯一需要做的就是，他没有给你数据库的GUI(如果这个再有的话，emmmmmm一言难尽)

你需要自己打开数据库的命令行来创建表：

![](https://ws1.sinaimg.cn/large/6af92b9fgy1fp2cduezmoj20kw0a8aag.jpg)

> 免费的用户不能使用SSH链接，也就意味着免费试用者智能在Web端访问他们的bash，数据库和下面将要说道的服务器bash都是一样的。 这样的缺点是网页的console特别慢。。。。。

数据库搞定，访问提供的免费域名,emmmm出错了，不过错误日志很方便查看：

![](https://ws1.sinaimg.cn/large/6af92b9fgy1fp2cs60kssj20ic03uaa0.jpg)

emmmm  没有pymysql，前面刚说了不用再搭环境了，这一秒打脸？？？

好吧，好在官网给出了解决方案，就是在网页打开服务器bash，你自己手动装一个....pip就OK...

好了 这下可以了：

![](https://ws1.sinaimg.cn/large/6af92b9fgy1fp2cx5luz6j20a50a6q35.jpg)


## 小结
说说作者的这套技术路线吧

算法 + 系统 也是我的专业最好的结合方式，这里作者给出了一个初步的web解决方案，只使用一种语言、一种工具。总的来讲是很棒的。但是有一个大问题就是学习器的加速，整个框架文件学习器体积最大，io效率最慢。如果要更变参数又得重新学习，时间成本很大。如果未来机器学习算法程序真的要面向用户，我认为作者提出的此套技术路径是不太适合大型项目的。