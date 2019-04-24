# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:46:31 2019

@author: daksh
"""

# Command for runnning via IPython console in Spyder:
# runfile('mnist_2.py')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle

np.random.seed(138)

""" For testing only start"""
num_epochs = 500
""" For testing only end"""

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

X = X_train
Y = Y_train

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

def compute_multiclass_loss(Y, Y_hat):
    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[1]
    L = -(1/m) * L_sum

    return L

def init_nnprams():
    n_x = X_train.shape[0]
    n_h = 64
    alpha = 1
    
    W1 = np.random.randn(n_h, n_x)
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(digits, n_h)
    b2 = np.zeros((digits, 1))
    
    nn_params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    
    return nn_params, alpha
    

def train_save(nn_params, num_epochs, alpha, X, Y):
    print('Start Training')
    W1 = nn_params['W1']
    b1 = nn_params['b1']
    W2 = nn_params['W2']
    b2 = nn_params['b2']
    
    for i in range(num_epochs):
    
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
    
        W2 = W2 - alpha * dW2
        b2 = b2 - alpha * db2
        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1
    
        if (i % 100 == 0):
            print("Epoch", i, "cost: ", cost)
    
    print("Training Complete, Final cost:", cost)
    nn_params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    with open('nn_params.pkl', 'wb') as f:
        pickle.dump(nn_params, f)
    f.close()
    np.save('nn_params.npy', nn_params)
    
    return nn_params

def ann_test(nn_params, X_test):
    W1 = nn_params['W1']
    b1 = nn_params['b1']
    W2 = nn_params['W2']
    b2 = nn_params['b2']
    Z1 = np.matmul(W1, X_test) + b1
    A1 = sigmoid(Z1)
    Z2 = np.matmul(W2, A1) + b2
    A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=0)
    
    return A2

while(1):
    choice_train = input('Enter 1 for training ANN or 2 for loading pretrained parameters: ')
    choice_train = int(choice_train)
    if choice_train != 1 and choice_train != 2:
        print('Enter 1 or 2 only: ')
        pass
    elif choice_train == 1:
        num_epochs = input('Enter number of epochs to train the ANN for: ')
        num_epochs = int(num_epochs)
        nn_params, alpha = init_nnprams()
        nn_params = train_save(nn_params, num_epochs, alpha, X, Y)
        break
    elif choice_train == 2:
        with open('nn_params.pkl', 'rb') as pickle_file:
            nn_params = pickle.load(pickle_file)
        break

num_predict = input('Enter a number for ANN to classify [0-9] ')

y_test_labels = y[0,m:]

idx_test = np.where(y_test_labels == int(num_predict))
X_test_1 = X_test[:,idx_test[0][0]].reshape(784,1)
out_test = ann_test(nn_params, X_test_1)
pred_num = np.around(np.max(out_test)*100, decimals=2)
pred_num_idx = np.argmax(out_test)
str_text = 'Predicted number is ' + str(pred_num_idx) + ' with ' + str(pred_num) + '% accuracy'

plt.imshow(X_test_1.reshape(28,28), cmap = matplotlib.cm.binary)
plt.axis("off")
plt.title(str_text)
plt.show()






