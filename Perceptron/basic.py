import numpy as np 
class perceptron():
    def __init__(self,eta = 0.01,n_iter = 50,random_state = 100):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    def fit(self,X,y):
        self.w = np.random.RandomState(self.random_state).normal(loc=0,scale=0.01,size= X.shape[1]+1)
        self.error = []
        for _ in range(self.n_iter):
            errors = 0
            for xi,target in zip(X,y):
                update = self.eta*(target - self.predcit(xi))
                self.w[1:] += update * xi
                self.w[0]  = update
                errors += int(update!=0)
            self.error.append(errors)
        return self
    def net_input(self,xi):
        return np.dot(xi,self.w[1:]) + self.w[0]
    def predcit(self,xi):
        return np.where(self.net_input(xi)>= 0.0,1,-1)
        