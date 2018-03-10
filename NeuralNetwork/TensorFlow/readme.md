The individual steps for building and compiling such a computation
graph in TensorFlow are as follows:
- Instantiate a new, empty computation graph.
- Add nodes (tensors and operations) to the computation graph.
- Execute the graph:
    - Start a new session
    - Initialize the variables in the graph
    - Run the computation graph in this session

> Furthermore, there is a universal way
of running both tensors and operators: tf.Session().run(). Using this method, as
we'll see later on as well, multiple tensors and operators can be placed in a list or
tuple. As a result, tf.Session().run() will return a list or tuple of the same size.
sess.run()只能运行operator和Tensor

## placeholder
tensorflow里的占位符，需用到时需传入dict

不指定样本个数的rank2：
```python
with g.as_default():
... tf_x = tf.placeholder(tf.float32,
... shape=[None, 2],
... name='tf_x')
```
## Variables in TensorFlow
In the context of TensorFlow, variables are a special type of tensor objects that allow
us to store and update the parameters of our models in a TensorFlow session during
training.

在执行node之前，variable必须被初始化
```python
with g.as_default():
    # 创建权重Node:
    w = tf.Variables(np.array([[1,2,3],[4,5,6]]),name = 'w')
    # 此时创建完成后，并不会分配内存（初始化）
with tf.Session(graph=g) as sess:
    # tf.glabal_variables_initializer()返回一个operator 会全局初始化参数
    sess.run(tf.glabal_variables_initializer())
    sess.run(w)
```
## Variable Scope
With variable scopes, we can organize the variables into separate subparts
```python
with g.as_default():
... with tf.variable_scope('net_A'):
...     with tf.variable_scope('layer-1'):
...         w1 = tf.Variable(tf.random_normal(
...             shape=(10,4)), name='weights')
...     with tf.variable_scope('layer-2'):
...         w2 = tf.Variable(tf.random_normal(
...                 shape=(20,10)), name='weights')
... with tf.variable_scope('net_B'):
...     with tf.variable_scope('layer-1'):
...         w3 = tf.Variable(tf.random_normal(
...             shape=(10,4)), name='weights')
```
## Visualizing the graph with TensorBoard
实现一个Variable Scope示例代码的可视化：
```python
def build_classifier(data, labels, n_classes=2):
    data_shape = data.get_shape().as_list()
    weights = tf.get_variable(name='weights',
                              shape=(data_shape[1], n_classes),
                              dtype=tf.float32)
    bias = tf.get_variable(name='bias', 
                           initializer=tf.zeros(shape=n_classes))
    print(weights)
    print(bias)
    logits = tf.add(tf.matmul(data, weights), 
                    bias, 
                    name='logits')
    print(logits)
    return logits, tf.nn.softmax(logits)

def build_generator(data, n_hidden):
    data_shape = data.get_shape().as_list()
    w1 = tf.Variable(
        tf.random_normal(shape=(data_shape[1], 
                                n_hidden)),name='w1')
    b1 = tf.Variable(tf.zeros(shape=n_hidden),
                     name='b1')
    hidden = tf.add(tf.matmul(data, w1), b1, 
                    name='hidden_pre-activation')
    hidden = tf.nn.relu(hidden, 'hidden_activation')
        
    w2 = tf.Variable(
        tf.random_normal(shape=(n_hidden, 
                                data_shape[1])),
        name='w2')
    b2 = tf.Variable(tf.zeros(shape=data_shape[1]),
                     name='b2')
    output = tf.add(tf.matmul(hidden, w2), b2, 
                    name = 'output')
    return output, tf.nn.sigmoid(output)
    
########################
## Defining the graph ##
########################

batch_size=64
g = tf.Graph()

with g.as_default():
    tf_X = tf.placeholder(shape=(batch_size, 100),dtype=tf.float32,name='tf_X')
    ## build the generator
    with tf.variable_scope('generator'):
        gen_out1 = build_generator(data=tf_X, 
                                   n_hidden=50)
    
    ## build the classifier
    with tf.variable_scope('classifier') as scope:
        ## classifier for the original data:
        cls_out1 = build_classifier(data=tf_X, 
                                    labels=tf.ones(
                                        shape=batch_size))
        
        ## reuse the classifier for generated data
        scope.reuse_variables()
        cls_out2 = build_classifier(data=gen_out1[1],
                                    labels=tf.zeros(
                                        shape=batch_size))
        
        init_op = tf.global_variables_initializer()
```
会话需要执行：
```python
with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('E:/logs',graph=g)
    writer.close()
```
然后cmd执行：
```bash
tensorboard --logdir=E:/logs
```
windows中有一坑，如果你存储log的路径中含有中文，网页是不会显示的

![](https://ws1.sinaimg.cn/large/6af92b9fgy1fp80jqfaxsj20z50g540f.jpg)