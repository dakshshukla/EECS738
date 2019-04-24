# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 10:56:20 2019

@author: daksh
"""

# Command for runnning via IPython console in Spyder:
# runfile('mnist_1.py', args = '500')
# arguments = number of epochs to train for

import numpy as np
import matplotlib.pyplot as plt
import sys

""" For testing only start"""
#num_epochs = 500
""" For testing only end"""

# Input arguments:
num_epochs = int(sys.argv[1])

X = np.load('images_flat.npy')
y = np.load('labels.npy')
X = X / 255

y_new = np.zeros(y.shape)
y_new[np.where(y == 0.0)[0]] = 1
y = y_new

m = 60000
m_test = X.shape[0] - m
X_train = X[:m].T
X_test = X[m:].T
y_train = y[:m].reshape(1,m)
y_test = y[m:].reshape(1,m_test)

np.random.seed(138)
shuffle_index = np.random.permutation(m)
X_train, y_train = X_train[:,shuffle_index], y_train[:,shuffle_index]

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

def compute_loss(Y, Y_hat):
    m = Y.shape[1]
    L = -(1./m) * ( np.sum( np.multiply(np.log(Y_hat),Y) ) + np.sum( np.multiply(np.log(1-Y_hat),(1-Y)) ) )

    return L

learning_rate = 1

X = X_train
Y = y_train

n_x = X.shape[0]
m = X.shape[1]

W = np.random.randn(n_x, 1) * 0.01
b = np.zeros((1, 1))

print('Start Training')

for i in range(num_epochs):
    
    Z = np.matmul(W.T, X) + b
    A = sigmoid(Z)

    cost = compute_loss(Y, A)

    dW = (1/m) * np.matmul(X, (A-Y).T)
    db = (1/m) * np.sum(A-Y, axis=1, keepdims=True)

    W = W - learning_rate * dW
    b = b - learning_rate * db

    if (i % 100 == 0):
        print("Epoch", i, "cost: ", cost)

print("Final cost:", cost)

Z = np.matmul(W.T, X_test) + b
A = sigmoid(Z)

#i = 3
#plt.imshow(X_train[:,i].reshape(28,28), cmap = matplotlib.cm.binary)
#plt.axis("off")
#plt.show()

idx_0 = np.where(y_test == 1)

Accuracy = np.sum(1 - (y_test[idx_0] - A[idx_0]))/A[idx_0].shape[0]
s = 'Accuracy = ' + np.str(np.around(Accuracy, decimals = 4)*100) + '%'

plt.plot(np.arange(A[idx_0].shape[0]), A[idx_0], 'b.')
plt.plot(np.arange(A[idx_0].shape[0]), y_test[idx_0], 'r.')
plt.xlabel('Data Points')
plt.ylabel('Class 0 or 1')
plt.rcParams.update({'font.size': 12})
plt.legend(['Predicted Class', 'True Class'])
plt.title('ANN Predicted vs True Class for number \'0\'')
plt.ylim([0, 1.5])
plt.text(10, 1.25, s, fontsize=12)
plt.show()


