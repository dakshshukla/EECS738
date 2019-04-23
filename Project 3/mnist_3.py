# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:46:31 2019

@author: daksh
"""

# Command for runnning via IPython console in Spyder:
# runfile('mnist_2.py', args = '500')
# arguments = number of epochs to train for


import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib

np.random.seed(138)

""" For testing only start"""
num_epochs = 500
""" For testing only end"""

# Input arguments:
#num_epochs = int(sys.argv[1])

X = np.load('images_flat.npy')
y = np.load('labels.npy')
X = X / 255

digits = 10
examples = y.shape[0]

y = y.reshape(1, examples)

Y_new = np.eye(digits)[y.astype('int32')]
Y_new = Y_new.T.reshape(digits, examples)

m = 60000
m_test = X.shape[0] - m

X_train, X_test = X[:m].T, X[m:].T
Y_train, Y_test = Y_new[:,:m], Y_new[:,m:]

shuffle_index = np.random.permutation(m)
X_train, Y_train = X_train[:, shuffle_index], Y_train[:, shuffle_index]

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

def compute_multiclass_loss(Y, Y_hat):

    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[1]
    L = -(1/m) * L_sum

    return L

n_x = X_train.shape[0]
n_h = 64
learning_rate = 1

W1 = np.random.randn(n_h, n_x)
b1 = np.zeros((n_h, 1))
W2 = np.random.randn(digits, n_h)
b2 = np.zeros((digits, 1))

X = X_train
Y = Y_train

print('Start Training')

for i in range(2000):

    Z1 = np.matmul(W1,X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.matmul(W2,A1) + b2
    A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=0)

    cost = compute_multiclass_loss(Y, A2)

    dZ2 = A2-Y
    dW2 = (1./m) * np.matmul(dZ2, A1.T)
    db2 = (1./m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.matmul(W2.T, dZ2)
    dZ1 = dA1 * sigmoid(Z1) * (1 - sigmoid(Z1))
    dW1 = (1./m) * np.matmul(dZ1, X.T)
    db1 = (1./m) * np.sum(dZ1, axis=1, keepdims=True)

    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1

    if (i % 100 == 0):
        print("Epoch", i, "cost: ", cost)

print("Training Complete, Final cost:", cost)

num_predict = input('Enter a number for ANN to classify [0-9] ')

y_test_labels = y[0,m:]

idx_test = np.where(y_test_labels == int(num_predict))

Z1 = np.matmul(W1, X_test[:,idx_test[0][0]]) + b1
A1 = sigmoid(Z1)
Z2 = np.matmul(W2, A1) + b2
A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=0)

_, axarr = plt.subplots(1,2,figsize=(10,10))

axarr[0].imshow(X_test[:,idx_test[0][0]].reshape(28,28), cmap = matplotlib.cm.binary)
plt.axis("off")
plt.show()
axarr[1].plot(1,1)
plt.text(str(A2))
print(y_test_labels[idx_test[0][0]])
