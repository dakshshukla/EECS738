# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 11:03:12 2019

@author: daksh
"""

# Command for runnning via IPython console in Spyder:
# runfile('airfoil_predict.py')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
import pandas as pd    

np.random.seed(138)

with open('airfoil_self_noise.dat','r') as f:
    df = pd.DataFrame(l.rstrip().split() for l in f)

df = df.values
df = df.astype(float)

""" For testing only start"""
num_epochs = 50
""" For testing only end"""

X_all = df[:,0:5]
y_all = df[:,5]

num_train_data = 1000
train_idx = np.random.permutation(num_train_data)

X_train, X_test = X_all[train_idx, :], X_all[train_idx, :]
Y_train, Y_test = y_all[train_idx, :], y_all[train_idx, :]

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

def loss_fun(Y, Y_hat):
    L_sum = np.sum(np.square(Y - Y_hat))
    m = Y.shape[1]
    L = -(1/m) * L_sum

    return L

def init_nnprams():
    n_x = X_train.shape[0]
    n_h = 64
    alpha = 5
    p = 0.7 # momentum
    mini_batch = 2000
    
    W1 = np.random.randn(n_h, n_x)*np.sqrt(1/n_x)
    b1 = np.zeros((n_h, 1))*np.sqrt(1/n_x)
    W2 = np.random.randn(digits, n_h)*np.sqrt(1/n_h)
    b2 = np.zeros((digits, 1))*np.sqrt(1/n_h)
    
    p_W1 = np.zeros(W1.shape)
    p_b1 = np.zeros(b1.shape)
    p_W2 = np.zeros(W2.shape)
    p_b2 = np.zeros(b2.shape)
    
    nn_params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'p_W1': p_W1, 'p_b1': p_b1, 'p_W2': p_W2, 'p_b2': p_b2}
    
    return nn_params, alpha, p, mini_batch

def train_save(nn_params, num_epochs, alpha, X_train, Y_train, mini_batch, p):
    print('Start Training')
    W1 = nn_params['W1']
    b1 = nn_params['b1']
    W2 = nn_params['W2']
    b2 = nn_params['b2']
    p_W1 = nn_params['p_W1']
    p_b1 = nn_params['p_b1']
    p_W2 = nn_params['p_W2']
    p_b2 = nn_params['p_b2']
    
    for i in range(num_epochs):
        
        extra_batch = m%mini_batch
        num_iter = int((m - extra_batch)/mini_batch);
        shuffle_index = np.random.permutation(m)
        X_train_shuf, Y_train_shuf = X_train[:, shuffle_index], Y_train[:, shuffle_index]
        
        for ii in range(num_iter):
            start_dp = ii*mini_batch
            end_dp = (ii+1)*mini_batch - 1
            X = X_train_shuf[:,start_dp:end_dp]
            Y = Y_train_shuf[:,start_dp:end_dp]

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
            
            p_W1 = p * p_W1 + (1 - p) * dW1
            p_b1 = p * p_b1 + (1 - p) * db1
            p_W2 = p * p_W2 + (1 - p) * dW2
            p_b2 = p * p_b2 + (1 - p) * db2
            
            W2 = W2 - alpha * p_W2
            b2 = b2 - alpha * p_b2
            W1 = W1 - alpha * p_W1
            b1 = b1 - alpha * p_b1

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
        nn_params, alpha, p, mini_batch = init_nnprams()
        nn_params = train_save(nn_params, num_epochs, alpha, X_train, Y_train, mini_batch, p)
        break
    elif choice_train == 2:
        with open('nn_params.pkl', 'rb') as pickle_file:
            nn_params = pickle.load(pickle_file)
        break

num_predict = input('Enter a number for ANN to classify [0-9] ')

y_test_labels = y_all[0,m:]

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






