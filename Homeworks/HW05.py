# The funtionality can be achieved by only using numpy library
# Below is some functions you may need during implementation
# You may do "import numpy as np" and use these methods in your own way
from numpy import dot
from numpy import sum
from numpy import zeros
from numpy import sign
import numpy as np

class kernel_perceptron:

    def __init__(self, T=1):
        self.k = self.kernal
        self.T = T
        self.missclassified = list()
        self.a = self.activate_boundary
        self.support_v = np.zeros()
        self.support_vy = np.zeros()

    def train(self, X, Y):
        N, D = X.shape
        Z = np.zeros((N, N))
        self.missclassified = np.zeros(N)
        for t in range(self.T):
            for i in range(N):
                if np.sign(np.sum(Z[:,i] * self.missclassified * Y)) != Y[i]:
                    self.missclassified[i] += 1.0
        for i in range(N):
            for j in range(N):
                Z[i,j] = self.k(X[i], X[j])
        
        support_v = self.missclassified > 0
        
        ind = np.arange(len(self.missclassified))[support_v]
        self.missclassified = self.missclassified[support_v]
        
        self.support_v = X[support_v]
        self.support_vy = Y[support_v]
        print (len(self.missclassified),N)

    def project(self, X):
        y_predict = np.zeros(len(X))
        for i in range(len(X)):
            kernal_count = 0
            for a, support_vy, support_v in zip(self.missclassified, self.support_vy, self.support_v):
                kernal_count += a * support_vy * self.k(X[i], support_v)
            y_predict[i] = kernal_count
        return y_predict

    def predict(self, X):
        X = np.atleast_2d(X)
        N, D = X.shape
        '''
        if(self.project(X) == 0):
            return 0
        if(self.project(X) < 0):
            return -1
        else:
            return 1
        '''
        return np.sign(self.project(X))
    
    def kernal(x, Y, p=3):
        return (1 + np.dot(x, Y)) ** p
    
    def activate_boundary(self, x):
        return np.where(x<=0.0, -1, 1)
