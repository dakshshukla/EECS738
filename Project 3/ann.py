# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 15:51:50 2019

@author: daksh
"""
import numpy as np

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],4)
        self.weights2   = np.random.rand(self.weights1.shape[1],1)
        self.y          = y
        self.output     = np.zeros(y.shape)
        
    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
        
    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (-2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(-2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))
        # update the weights with the derivative (slope) of the loss function
        self.weights1 -= d_weights1
        self.weights2 -= d_weights2
        
        return self.weights1, self.weights2

if __name__ == "__main__":
    X = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])
    y = np.array([[0],[1],[1],[0]])
    nn = NeuralNetwork(X,y)

    for i in range(2000):
        nn.feedforward()
        w1, w2 = nn.backprop()
    
    pred_out = nn.output
    print('True value = ' + str(y[0,0]) + ', Predicted value = ' + str(pred_out[0,0]))
    print('True value = ' + str(y[1,0]) + ', Predicted value = ' + str(pred_out[1,0]))
    print('True value = ' + str(y[2,0]) + ', Predicted value = ' + str(pred_out[2,0]))
    print('True value = ' + str(y[3,0]) + ', Predicted value = ' + str(pred_out[3,0]))
