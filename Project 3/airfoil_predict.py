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
m = len(df)

train_idx = np.random.rand(m) < 0.9

X_train, X_test = X_all[train_idx, :].T, X_all[~train_idx, :].T
Y_train, Y_test = y_all[train_idx].T, y_all[~train_idx].T

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

def loss_fun(Y, Y_hat):
    L = np.mean((Y - Y_hat)**2)

    return L

def init_nnprams():
    n_x = X_train.shape[0]
    n_h = 10
    alpha = 0.1
    p = 0.8 # momentum
    mini_batch = 1500
    
    W1 = np.random.randn(n_h, n_x)*np.sqrt(1/n_x)
    b1 = np.zeros((n_h, 1))*np.sqrt(1/n_x)
    W2 = np.random.randn(1, n_h)*np.sqrt(1/n_h)
    b2 = np.zeros((1, 1))*np.sqrt(1/n_h)
    
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
        num_iter = int((m - extra_batch)/mini_batch)
        train_size = X_train.shape[1]
        shuffle_index = np.random.permutation(train_size)
        X_train_shuf, Y_train_shuf = X_train[:,shuffle_index], Y_train[shuffle_index]
        
        for ii in range(num_iter):
            start_dp = ii*mini_batch
            end_dp = (ii+1)*mini_batch - 1
            X = X_train_shuf[:,start_dp:end_dp]
            Y = Y_train_shuf[start_dp:end_dp]

            Z1 = np.matmul(W1,X) + b1
            A1 = sigmoid(Z1)
            Z2 = np.matmul(W2,A1) + b2
            A2 = Z2
        
            cost = loss_fun(Y, A2)
            
            data_size = X.shape[1]
            
            db2 = 2*(A2 - Y)
            dW2 = ((1/data_size)*np.matmul(A1, db2.T)).T
            db2 = (1/data_size)*np.sum(db2)
            
            dA1 = np.matmul(W2.T, 2*(A2 - Y))
            dZ1 = dA1*sigmoid(Z1)*(1 - sigmoid(Z1))
            dW1 = (1/data_size)*np.matmul(dZ1, X.T)
            db1 = (1/data_size)*np.sum(dZ1)
            
            p_W1 = p * p_W1 + (1 - p) * dW1
            p_b1 = p * p_b1 + (1 - p) * db1
            p_W2 = p * p_W2 + (1 - p) * dW2
            p_b2 = p * p_b2 + (1 - p) * db2
            
            W2 = W2 - alpha * p_W2
            b2 = b2 - alpha * p_b2
            W1 = W1 - alpha * p_W1
            b1 = b1 - alpha * p_b1

        if i%50 == 0:
            print("Epoch", i, "cost: ", cost)
    
    print("Training Complete, Final cost:", cost)
    nn_params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    with open('nn_params_airfoil.pkl', 'wb') as f:
        pickle.dump(nn_params, f)
    f.close()
    
    return nn_params

def ann_test(nn_params, X_test):
    W1 = nn_params['W1']
    b1 = nn_params['b1']
    W2 = nn_params['W2']
    b2 = nn_params['b2']
    Z1 = np.matmul(W1, X_test) + b1
    A1 = sigmoid(Z1)
    Z2 = np.matmul(W2, A1) + b2
    A2 = Z2
    
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
        with open('nn_params_airfoil.pkl', 'rb') as pickle_file:
            nn_params = pickle.load(pickle_file)
        break

out_test = ann_test(nn_params, X_test)

error = Y_test - out_test
mse = np.mean(np.sqrt(error**2))
print('MSE: ', mse)



