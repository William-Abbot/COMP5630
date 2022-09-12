#William Abbot
#mini project 2

import numpy as np
import re
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd



data_dir = 'mnist.scale\\'
file = open(data_dir+'mnist.scale.txt')
labels = list()
data = list()
regex = re.compile(r'(\d+)(:)(\d)(.)(\d+)')
regex1 = re.compile(r'(\d+)(:)(\d)')
lines = file.readlines()

for line in lines[:30000]:
    temp_data = [0]*784
    label = line[:1]
    labels.append(float(label))
    line = line[2:]
    space_counter = 0
    in_between_space = ''
    for s in range(len(line)):
        if line[s] == ' ':
            space_counter += 1
        else:
            in_between_space += line[s]
        if space_counter == 1:
            space_counter = 0
            if ':1' in in_between_space:
                mo = regex1.search(in_between_space)
                feat = int(mo.group(1))
                x_i = float(mo.group(3))
                in_between_space = ''
            else:
                mo = regex.search(in_between_space)
                #print(in_between_space)
                feat = int(mo.group(1))
                x_i = float(mo.group(3) + mo.group(4) + mo.group(5))
                #print(x_i)
                in_between_space = ''
            #print(feat)
            temp_data[feat-1] = x_i
            #print(x_i)
    data.append(temp_data)

data = np.array(data)
labels = np.array(labels)



class neuralNet:

    def __init__(self, sizes, e=100, l_rate=0.03):
        self.sizes = sizes
        self.e = e
        self.l_rate = l_rate
        self.params = self.initialization()

    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))

    def softmax(self, x, derivative=False):
        exps = np.exp(x - x.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)

    def initialization(self):
        input_layer=self.sizes[0]
        hidden_1=self.sizes[1]
        output_layer=self.sizes[2]

        params = {
            'W1':np.random.randn(hidden_1, input_layer) * np.sqrt(1. / hidden_1),
            'W2':np.random.randn(output_layer, hidden_1) * np.sqrt(1. / output_layer)
        }

        return params

    def forward_pass(self, x_train):
        params = self.params
        params['A0'] = x_train
        params['Z1'] = np.dot(params["W1"], params['A0'])
        params['A1'] = self.sigmoid(params['Z1'])
        params['Z2'] = np.dot(params["W2"], params['A1'])
        params['A2'] = self.softmax(params['Z2'])

        return params['A2']

    def backward_pass(self, y_train, output):

        params = self.params
        change_w = {}
        error = 2 * (output - y_train) / output.shape[0] * self.softmax(params['Z2'], derivative=True)
        change_w['W2'] = np.outer(error, params['A1'])
        error = np.dot(params['W2'].T, error) * self.sigmoid(params['Z1'], derivative=True)
        change_w['W1'] = np.outer(error, params['A0'])

        return change_w

    def update_network_parameters(self, changes_to_w):

        
        for key, value in changes_to_w.items():
            self.params[key] -= self.l_rate * value

    def compute_accuracy(self, x_val, y_val):

        predictions = []

        for x, y in zip(x_val, y_val):
            output = self.forward_pass(x)
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(y))
        return np.mean(predictions)

    def train(self, x_train, y_train, x_val, y_val):
        start_time = time.time()
        accuracy = 0
        average = 0
        for iteration in range(self.e):
            for x,y in zip(x_train, y_train):
                output = self.forward_pass(x)
                changes_to_w = self.backward_pass(y, output)
                self.update_network_parameters(changes_to_w)
            
            accuracy = self.compute_accuracy(x_val, y_val)
            average += accuracy
        
        return ('average NN testing accuracy is: ' + str(average/self.e) + '%')


class kernelizedPerceptron:

    def __init__(self, hp=1):
        self.hp = hp
        self.mistake_counter = np.zeros(1)
        self.a = self.activation
        self.support_v = np.zeros(1)
        self.support_vy = np.zeros(1)

    def train(self, X, y):
        #assert(type(X) == np.ndarray)
        N, D = X.shape
        self.mistake_counter = np.zeros(N, dtype=np.float64)

        K_arr = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if np.sign(np.sum(self.kernel(X[i], X[j]) * self.mistake_counter[i] * y[i])) != y[j]:
                    self.mistake_counter[i] += 1.0
        
        sv = self.mistake_counter > 0
        self.mistake_counter = self.mistake_counter[sv]
        self.support_v = X[sv]
        self.support_vy = y[sv]
        
    def project(self, X):
        y_predict = np.zeros(len(X))
        for i in range(len(X)):
            kernal_count = 0
            for mc, vy, v in zip(self.mistake_counter, self.support_vy, self.support_v):
                kernal_count += mc * vy * self.kernel(X[i], v)
            y_predict[i] = kernal_count
        return y_predict

    def predict(self, X):
        #print(X)
        X = np.atleast_2d(X)
        #print(X.shape)
        N, D = X.shape
        return self.a(self.project(X))
    
    def kernel(self, X1, X2):
        return (1 + np.dot(X1, X2))**3
    
    #basically the sgn() function
    def activation(self, x):
        if x < 0.0:
            return -1
        if x == 0.0:
            return 0
        if x > 0.0:
            return 1



class linearSVM:
    
    def __init__(self, hp1=0.001, hp2=0.07, N=1200):
        self.weight = None
        self.b = None
        self.lr = hp1
        self.lp = hp2
        self.max_passes = N
    

    def train(self, X, y):
        N, D = X.shape     
        sgnY = np.array(np.sign(y))        
        self.weight = np.zeros(D)
        self.bias = 0
        
        counter = 0
        while counter < self.max_passes:
            for i, x in enumerate(X):
                positive = sgnY[i]*(np.dot(x, self.weight) - self.bias) >= 1.0
                if positive:
                    # if classification is correct, just subtract weight prop. to hyper param
                    self.weight -= self.lr * (2 * self.lp * self.weight)
                else:
                    # if classification is incorrect, subtract prop. to hp and by dot product of x and sign of labels
                    self.weight -= self.lr * (2 * self.lp * self.weight - np.dot(x, sgnY[i]))
                    self.bias -= self.lr * sgnY[i]
            counter += 1

    def predict(self, X):
        approx = np.dot(X, self.weight) - self.bias
        return np.sign(approx)

bits = 4

x_train = data[:21000]
y_train = labels[:21000]
x_test = data[2100:]
y_test = labels[2100:]

y_train = ["{0:04b}".format(int(y)) for y in y_train]
for index in range(len(y_train)):
    y_train[index] = [int(d) for d in y_train[index]]

y_train = np.array(y_train)
y_trainTemp = [np.copy(y_train[:,b]) for b in range(bits)]

y_trainTemp = np.array(y_trainTemp)






perceptron = [0]*4
perceptron[0] = kernelizedPerceptron()
perceptron[1] = kernelizedPerceptron()
perceptron[2] = kernelizedPerceptron()
perceptron[3] = kernelizedPerceptron()


for index in range(bits):
    perceptron[index].train(x_train, y_trainTemp[index])

y_hat = np.zeros(len(y_test))

for y in range(len(y_test)):
    prediction = [0]*4
    prediction[0] = perceptron[0].predict(x_test[y])
    prediction[1] = perceptron[1].predict(x_test[y])
    prediction[2] = perceptron[2].predict(x_test[y])
    prediction[3] = perceptron[3].predict(x_test[y])
    y_hat[y]= np.array(int("".join(str(x) for x in prediction), 2))
 

print(len(y_hat), ' test case predicted.', sep='')
correct_num = sum(y_hat == y_test)
print(correct_num, ' are correct.', sep='')
print('\n\nPerceptron testing Accuracy = ', np.round(correct_num * 100 / len(y_hat)), '%', sep='')

confusion_matrix(y_test, y_hat)







svm = [linearSVM() for kp in range(bits)]

for index in range(bits):

    svm[index].train(x_train, y_trainTemp[index])

y_hat = np.zeros(len(y_test))

for y in range(len(y_test)):
    prediction = [0]*4
    prediction[0] = svm[0].predict(x_test[y])
    prediction[1] = svm[1].predict(x_test[y])
    prediction[2] = svm[2].predict(x_test[y])
    prediction[3] = svm[3].predict(x_test[y])
    y_hat[y]= np.array(int("".join(str(int(x)) for x in prediction), 2))
    
correct_num = sum(y_hat == y_test)
print('\n\nsvm testing Accuracy = ', np.round(correct_num * 100 / len(y_hat)), '%', sep='')
print('\n')



def to_categorical(y, num_classes=None, dtype='float32'):

  y = np.array(y, dtype='int')
  input_shape = y.shape
  if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
    input_shape = tuple(input_shape[:-1])
  y = y.ravel()
  if not num_classes:
    num_classes = np.max(y) + 1
  n = y.shape[0]
  categorical = np.zeros((n, num_classes), dtype=dtype)
  categorical[np.arange(n), y] = 1
  output_shape = input_shape + (num_classes,)
  categorical = np.reshape(categorical, output_shape)
  return categorical





x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.15, random_state=21)

myNN = neuralNet([784,100,10])

print(myNN.train(x_train, y_train, x_val, y_val))
