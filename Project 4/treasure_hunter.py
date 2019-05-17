# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 16:31:08 2019

@author: daksh
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import random

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD , Adam, RMSprop
from keras.layers.advanced_activations import PReLU

random.seed(3245)
np.random.seed(7895)

#tr_map = np.random.randint(2, size=(10,10))

# 1 = free space (white)
# 0 = obstacle (black)
def init_map():
    tr_map = np.array([
        [ 1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
        [ 1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
        [ 1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
        [ 0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  1.,  1.],
        [ 1.,  1.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  1.],
        [ 1.,  1.,  0.,  1.,  0.,  1.,  1.,  1.,  1.,  1.],
        [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
        [ 1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
        [ 1.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.],
        [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.]
    ])
    tr_map[0,0] = 0.5
    return tr_map

def plot_map(tr_map, pause_time):
    plt.figure(1)
    plt.grid('on')
    nrows, ncols = tr_map.shape
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, nrows, 1))
    ax.set_yticks(np.arange(0.5, ncols, 1))
    plt.title('Init Map, top left corner = start, bottom right corner = end, white = free space, black = obstacle')
    plt.imshow(tr_map, cmap='gray')
    plt.pause(pause_time)

def valid_actions(tr_map_state):
    actions = [0,1,2,3]
    idx_hunt = np.where(tr_map_state == 0.5)
    
    if idx_hunt[0] == 0: # cannot go up
        actions.remove(1)
    elif idx_hunt[0] == (tr_map_state.shape[0] - 1): # cannot go down
        actions.remove(3)
    
    if idx_hunt[1] == 0: # cannot go left
        actions.remove(0)
    elif idx_hunt[1] == (tr_map_state.shape[1] - 1): # cannot go right
        actions.remove(2)
    
    if idx_hunt[0] > 0: # not in top row
        if tr_map_state[idx_hunt[0]-1, idx_hunt[1]] == 0: # cannot go up
            actions.remove(1)
    if idx_hunt[0] < tr_map_state.shape[0]-1: # not in bottom row
        if tr_map_state[idx_hunt[0]+1, idx_hunt[1]] == 0: # cannot go down
            actions.remove(3)

    if idx_hunt[1] > 0: # not in left most column
        if tr_map_state[idx_hunt[0], idx_hunt[1]-1] == 0: # cannot go left
            actions.remove(0)
    if idx_hunt[1] < tr_map_state.shape[1]-1: # not in right most column
        if tr_map_state[idx_hunt[0], idx_hunt[1]+1] == 0: # cannot go right
            actions.remove(2)
    
    return actions

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

def loss_fun(Y, Y_hat):
    L = np.mean((Y - Y_hat)**2)
    return L

def init_nnprams(X_train):
    n_x = X_train.shape[0] + 1
    n_h = 100
    alpha = 0.01
    
#    W1 = np.random.randn(n_h, n_x)*np.sqrt(1/n_x)
    W1 = np.random.randn(n_h, n_x)
    b1 = np.zeros((n_h, 1))
#    W2 = np.random.randn(1, n_h)*np.sqrt(1/n_h)
    W2 = np.random.randn(1, n_h)
    b2 = np.zeros((1, 1))
    
    nn_params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    
    return nn_params, alpha

def train_save(nn_params, num_epochs, alpha, X_train, Y_train):
#    print('Start Training')
    W1 = nn_params['W1']
    b1 = nn_params['b1']
    W2 = nn_params['W2']
    b2 = nn_params['b2']
    
    for i in range(num_epochs):
        train_size = X_train.shape[1]
        shuffle_index = np.random.permutation(train_size)
        X_train_shuf, Y_train_shuf = X_train[:,shuffle_index], Y_train[:,shuffle_index]
        
        X = X_train_shuf
        Y = Y_train_shuf

        Z1 = np.matmul(W1,X) + b1
        A1 = sigmoid(Z1)
        Z2 = np.matmul(W2,A1) + b2
        if i == 0:
            start_cost = loss_fun(Y, Z2)
        cost = loss_fun(Y, Z2)
     
        db2 = (-2/train_size)*np.sum((Y - Z2), axis = 1, keepdims=True)
        dW2 = (-2/train_size)*np.matmul((Y - Z2), A1.T)
        
        dA1 = np.matmul(W2.T, 2*(Z2 - Y))
        dZ1 = dA1*sigmoid(Z1)*(1 - sigmoid(Z1))
        dW1 = (1/train_size)*np.matmul(dZ1, X.T)
        db1 = (1/train_size)*np.sum(dZ1)
        
        W2 = W2 - alpha * dW2
        b2 = b2 - alpha * db2
        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1

#        if i%5 == 0:
#            print("Epoch", i, "cost: ", cost)
    
    print('Training Complete, Start Cost: ', start_cost, 'Final cost: ', cost)
    nn_params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
#    with open('nn_params.pkl', 'wb') as f:
#        pickle.dump(nn_params, f)
#    f.close()
    
    return nn_params

def build_model(lr=0.001):
    model = Sequential()
    model.add(Dense(101, input_shape=(101,)))
    model.add(PReLU())
    model.add(Dense(101))
    model.add(PReLU())
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def ann_predict(nn_params, X_test):
    W1 = nn_params['W1']
    b1 = nn_params['b1']
    W2 = nn_params['W2']
    b2 = nn_params['b2']
    Z1 = np.matmul(W1, X_test) + b1
    A1 = sigmoid(Z1)
    Z2 = np.matmul(W2, A1) + b2
    return Z2

def reward_fun(tr_map_state0, action, tr_map_state1):
    if action not in valid_actions(tr_map_state0):
        reward = -1
    else:
        reward = -0.02
    idx_hunt1 = np.where(tr_map_state1 == hunt_mark)
    if tr_map_state0[idx_hunt1[0], idx_hunt1[1]] == 0.8:
        reward = -0.25
    if idx_hunt1 == (tr_map_state0.shape[0]-1, tr_map_state0.shape[1]-1):
        reward = +1
    return reward

def update_state(tr_map_state, action):
    tr_map_state_out = tr_map_state
    if action in valid_actions(tr_map_state):
        idx_hunt = np.where(tr_map_state == 0.5)
        if action == 0: # left
            idx_hunt_next = (idx_hunt[0], idx_hunt[1]-1)
        elif action == 1: # Up
            idx_hunt_next = (idx_hunt[0]-1, idx_hunt[1])
        elif action == 2: # Right
            idx_hunt_next = (idx_hunt[0], idx_hunt[1]+1)
        elif action == 3: # Down
            idx_hunt_next = (idx_hunt[0]+1, idx_hunt[1])
        
        tr_map_state_out[idx_hunt[0], idx_hunt[1]] = 0.8
        tr_map_state_out[idx_hunt_next[0], idx_hunt_next[1]] = 0.5
    return tr_map_state_out

def play_game(tr_map_state, nn_params, gamma, iteration, model, beta):
    game_flag = 0
    cum_reward = 0
    actions_all = [0,1,2,3]
    tr_map_state0 = tr_map_state.copy()
    while True:
        val_actions = valid_actions(tr_map_state0)
        q = np.zeros((4,1))
#        if np.random.rand() < (epsilon**iteration):
        if np.random.rand() < 0.1:
            action_apply = random.choice(val_actions)
#            print('Random: ', actions_dict[action_apply])
        else:
            for a in actions_all:
                a = np.array([[a]])
                ann_input = np.append(tr_map_state0.reshape((-1, 1)), a, axis = 0)
                q[a,0] = ann_predict(nn_params, ann_input)
#                q[a,0] = model.predict(ann_input.T)
            action_apply = np.argmax(q)
#            action_apply = np.argmax(ann_predict(nn_params, tr_map_state0.reshape((-1, 1))))
#            action_apply = np.argmax(model.predict(tr_map_state.reshape((1, -1))))
            #print('ANN: ', actions_dict[action_apply])
        tr_map_state0in = tr_map_state0.copy()
        tr_map_state1 = update_state(tr_map_state0in, action_apply)
        
#        q = ann_predict(nn_params, tr_map_state0.reshape((-1, 1)))
#        q = model.predict(tr_map_state0.reshape((1, -1)))
        
        reward = reward_fun(tr_map_state0in, action_apply, tr_map_state1)
#        q[action_apply,0] = reward + gamma*np.max(ann_predict(nn_params, tr_map_state1.reshape((-1, 1))))
#        q[0, action_apply] = reward + gamma*np.argmax(model.predict(tr_map_state1.reshape((1, -1))))
        Qa = np.zeros((4,1))
        for a in actions_all:
            tr_map_state1in = tr_map_state1.copy()
            tr_map_state2_try = update_state(tr_map_state1in, a)
#            q[a,0] = reward + gamma*np.max(ann_predict(nn_params, tr_map_state1_try.reshape((-1, 1))))
            a = np.array([[a]])
            ann_input = np.append(tr_map_state2_try.reshape((-1, 1)), a, axis = 0)
#            q[a,0] = reward + gamma*ann_predict(nn_params, ann_input)
#            q[a,0] = reward + gamma*model.predict(ann_input.T)
            Qa[a,0] = ann_predict(nn_params, ann_input)
#            Qa[a,0] = model.predict(ann_input.T)
        
        max_Qa = np.max(Qa)
        Q = np.max(q) + beta*( reward + gamma*max_Qa - np.max(q) )
        
        
        cum_reward = cum_reward + reward
#        cum_reward = cum_reward + q[0,action_apply]
        action_apply = np.array([[action_apply]])
        ann_input = np.append(tr_map_state0.reshape((-1, 1)), action_apply, axis = 0)
#        X_train = tr_map_state0.reshape((-1,1))
        X_train = ann_input
#        Y_train = np.array([[np.max(q)]])
        Y_train = np.array([[Q]])
#        Y_train = q.T
        
        if 'X_train_all' not in locals():
            X_train_all = X_train
            Y_train_all = Y_train
        else:
            X_train_all = np.append(X_train_all, X_train, axis = 1)
            Y_train_all = np.append(Y_train_all, Y_train, axis = 1)
        
        idx_hunt = np.where(tr_map_state1 == 0.5)
        if idx_hunt == (tr_map_state.shape[0]-1, tr_map_state.shape[1]-1):
            game_flag = 1
            print('Game Complete')
            break
        elif cum_reward <= -50:
            print('Reward Too Low')
            break
        elif X_train_all.shape[1] >= 500:
            print('100 pts collected')
            break
        
        tr_map_state0 = tr_map_state1.copy()
#        plot_map(tr_map_state0, pause_time)
#        print('Move: ', actions_dict[action_apply])
#        print('Reward: ', reward)
#        input('Press Enter to continue')
    return X_train_all, Y_train_all, game_flag, cum_reward, tr_map_state1

def test_ann(tr_map_state, model, nn_params):
    break_flag = 0
    actions_all = [0,1,2,3]
    tr_map_state0 = tr_map_state.copy()
    q = np.zeros((4,1))
    while True:
        for a in actions_all:
            a = np.array([[a]])
            ann_input = np.append(tr_map_state0.reshape((-1, 1)), a, axis = 0)
#            q[a,0] = model.predict(ann_input.T)
            q[a,0] = ann_predict(nn_params, ann_input)
        action_apply = np.argmax(q)
        
        tr_map_state0in = tr_map_state0.copy()
        tr_map_state1 = update_state(tr_map_state0in, action_apply)
        reward = reward_fun(tr_map_state0in, action_apply, tr_map_state1)
        idx_hunt = np.where(tr_map_state1 == 0.5)
        plot_map(tr_map_state1, pause_time)
        tr_map_state0 = tr_map_state1.copy()
        print('Move: ', actions_dict[action_apply])
        if idx_hunt == (tr_map_state.shape[0]-1, tr_map_state.shape[1]-1):
            print('ANN Game Complete')
            break_flag = 1
            break
        elif reward <= -100:
            print('ANN stuck')
            break
    return break_flag

visited_mark = 0.8  # Cells visited by the hunter will be painted by gray 0.8
hunt_mark = 0.5      # The current hunter cell will be painted by gray 0.5
epsilon = 0.99 # Exploration factor
pause_time = 1/30
tr_map = init_map()
#plot_map(tr_map, pause_time)
data_thresh = 1000
actions_dict = {0: 'left', 1: 'up', 2: 'right', 3: 'down'}
beta = 0.1
X_init = np.zeros((tr_map.reshape((-1,1))).shape)
nn_params, alpha = init_nnprams(X_init)
tr_map_state = tr_map
gamma = 0.95
num_epochs = 500
#X_train_all = np.zeros((tr_map.reshape((-1,1)).shape))
#Y_train_all = np.zeros((4,1))
iteration = 1
model = build_model()
while True:
#for i in range(0,10):
    game_flag = 0
    tr_map = init_map()
    tr_map_in = tr_map.copy()
    X_train, Y_train, game_flag, cum_reward, tr_map_out = play_game(tr_map_in, nn_params, gamma, iteration, model, beta)
    if game_flag == 1:
        print('Test ANN')
        break_flag = test_ann(tr_map_state, model, nn_params)
        if break_flag == 1:
            break
        else:
            game_flag = 0
    else:
        if iteration == 1:
            X_train_all = X_train
            Y_train_all = Y_train
        else:
            X_train_all = np.append(X_train_all, X_train, axis = 1)
            Y_train_all = np.append(Y_train_all, Y_train, axis = 1)
    _, X_train_all_idx = np.unique(X_train_all, return_index = True, axis = 1)
    X_train_all = X_train_all[:,X_train_all_idx]
    Y_train_all = Y_train_all[:,X_train_all_idx]
    if X_train_all.shape[1] > data_thresh:
        X_train_all = X_train_all[:,-data_thresh:]
        Y_train_all = Y_train_all[:,-data_thresh:]
    if X_train_all.shape[1] >= 10:
        train_size = round((X_train_all.shape[1]))
        shuffle_index = np.random.permutation(train_size)
        X_train_shuf, Y_train_shuf = X_train_all[:,shuffle_index], Y_train_all[:,shuffle_index]
        nn_params = train_save(nn_params, num_epochs, alpha, X_train_shuf, Y_train_shuf)
#        model.fit(
#                X_train_shuf.T,
#                Y_train_shuf.T,
#                epochs=100,
#                batch_size=100,
#                verbose=0,
#            )
    iteration = iteration + 1
    if iteration%1 == 0:
#        loss = model.evaluate(X_train_all.T, Y_train_all.T, verbose=0)
#        print('Loss: ', loss)
        plot_map(tr_map_out, pause_time)

def test_play():
    tr_map_state0 = init_map()
    while True:
        action = input('Enter action [0,1,2,3] OR 5 to quit: ')
        action = int(action)
        if action == 5:
            return False
        tr_map_state1 = tr_map_state0.copy()
        tr_map_state1 = update_state(tr_map_state1, action)
        reward = reward_fun(tr_map_state0, action, tr_map_state1)
        tr_map_state0 = tr_map_state1.copy()
        print('Move: ', actions_dict[action])
        print('Reward: ', reward)
        plot_map(tr_map_state1, pause_time)


test_ann(tr_map, model)

