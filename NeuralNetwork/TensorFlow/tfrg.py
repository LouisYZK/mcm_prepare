import tensorflow as tf 
import numpy as np
X_train = np.arange(10).reshape((10,1))
y_train = np.array([1.0, 1.3, 3.1,2.0, 5.0, 6.3,6.6, 7.4, 8.0,9.0])
class Tflinreg():
    def __init__(self,x_dim,learning_rate=0.01,random_seed=None):
        self.x_dim = x_dim
        self.learning_rate = learning_rate
        self.g = tf.Graph()
        with self.g.as_default():
            tf.set_random_seed(random_seed)
            self.build()
            self.init_op = tf.global_variables_initializer()
    def build(self):
        self.X = tf.placeholder(dtype=tf.float32,shape=(None,self.x_dim),name = 'x_input')
        self.y = tf.placeholder(dtype=tf.float32,shape = (None),name = 'y_input')
        w = tf.Variable(tf.zeros(shape=(1)),name = 'weight')
        b = tf.Variable(tf.zeros(shape=(1)),name = 'bias')
        self.z_net = tf.squeeze(w*self.X +b,name ='z_net')
        sqr_errors = tf.square(self.y - self.z_net,name = 'sqr_errors')
        self.mean_cost = tf.reduce_mean(sqr_errors,name = 'mean_cost')
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate,name = 'GradientDescent')
        self.optimizer = optimizer.minimize(self.mean_cost)
lrmodel = Tflinreg(x_dim=X_train.shape[1], learning_rate=0.01)
def train(sess,model,X_train,y_train,num_epochs = 10):
    sess.run(model.init_op)
    training_costs = []
    for i in range(num_epochs):
        _,cost = sess.run([model.optimizer,model.mean_cost],feed_dict={model.X:X_train,model.y:y_train})
        training_costs.append(cost)
    return training_costs
sess = tf.Session(graph=lrmodel.g)
training_costs = train(sess,lrmodel,X_train,y_train)
import matplotlib.pyplot as plt 
plt.plot(range(1,len(training_costs)+1),training_costs)
plt.tight_layout()
plt.xlabel('epoch')
plt.ylabel('Training Cost')
plt.show()

        