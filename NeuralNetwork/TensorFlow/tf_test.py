import tensorflow as tf 
g = tf.Graph()
with g.as_default():
    tf_a = tf.placeholder(tf.int32,shape=[],name='a')
    tf_b = tf.placeholder(tf.int32,shape=[],name='tf_b')
    z = tf_a*tf_b
with tf.Session(graph=g) as sess:
    feed_dict ={
        tf_a:1,
        tf_b:2
    }
    res = sess.run(z,feed_dict=feed_dict)
    print(res)