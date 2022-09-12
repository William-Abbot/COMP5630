# The funtionality can be achieved by only using numpy library
# Below is some functions you may need during implementation
# You may write your own import statement and use these methods in your own way
from numpy import exp
from numpy import log
from numpy import dot
from numpy import sum
from numpy import argmax
from numpy import ones
from numpy import hstack

from numpy.random import random_sample


class NeuralNetwork:
    def __init__(self):
        self.weights = self.initWeights()
        self.bias = 0
        self.H = 100
        self.K = 2
    
    def initWeights():
        w = list()
        for i in range(100*input_size):
            .append(random_sample)
    
    def sigmoid(self, x, derivative):
        if derivative:
            d = self.sigmoid(x, False)
            return d*(1-d)
        return 1/(1 + exp(-x))
    
    def fowardPass(x):
        pass
    
    def backProp():
        pass

    def train(X, y):
        pass