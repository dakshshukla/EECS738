# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 21:14:02 2019

@author: daksh
"""

# Command for runnning via IPython console in Spyder:
# runfile('EM_GMM.py', args = 'winequality-red.csv 1 5 50')
# runfile('EM_GMM.py', args = 'iris.csv 2 5 50')

import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, det

""" For testing only start"""
#data_select = np.asarray([[-0.39, 0.12, 0.94, 1.67, 1.76, 2.44, 3.72, 4.28, 4.92, 5.53, 0.06, 0.48, 1.01, 1.68, 1.80, 3.25, 4.12, 4.60, 5.28, 6.22]])
#data_select = np.reshape(data_select, (20,1))
#data_select = np.sort(data_select, axis = 0)
#fname = 'winequality-red.csv'
#fname = 'iris.csv'
#feature_num = 1
#max_K = 5
#max_epoch = 100
""" For testing only end"""

fname = sys.argv[1]
feature_num = int(sys.argv[2])
max_K = int(sys.argv[3])
max_epoch = int(sys.argv[4])

fdata = pd.read_csv(fname)
data_col = fdata.get_values()
data_select = np.reshape(data_col[:,feature_num-1], (np.size(data_col,0),1))
data_select = np.sort(data_select.astype(float), axis = 0)

eps_likelihood = 1e-3
likelihood_del = 1
best_run = 0

start_K = 2

def gauss_fun(x, mu, var):
    pd = (1/(np.sqrt(2*np.pi*var)))*np.exp(-1*np.square(x-mu)/(2*var))
    return pd

def em(max_K):
    for K in range (2, max_K+1):
        print('Trying #Gaussians or K =', K)
        epoch = 0
        while epoch < max_epoch:
            for mu_iter in range(0,K):
                idx = np.random.choice(data_select.shape[0], 1, replace=False)
                if mu_iter == 0:
                    idx_all = idx
                else:
                    idx_all = np.append(idx_all, idx)
            mu_hat = np.full((K,1), data_select[idx_all])
            var_hat = np.full((K,1), np.var(data_select))
            pi_hat = np.full((K,1), 1/K)
            gamma_hat = np.empty([np.size(data_select),K])
            iteration = 0
            likelihood_del = 1
            while(likelihood_del > eps_likelihood) and (iteration < 50):
                for k in range(0,K):
                    for n in range(0,np.size(data_select)):
                        # Expectation Step
                        numera_tor = (pi_hat[k,0])*(gauss_fun(data_select[n,0], mu_hat[k,0], var_hat[k,0]))
                        denomina_tor = np.sum((pi_hat[:,0])*(gauss_fun(data_select[n,0], mu_hat[:,0], var_hat[:,0])))
                        gamma_hat[n,k] = numera_tor/denomina_tor
                    # Maximization Step
                    N_k = np.sum(gamma_hat[:,k])
                    mu_hat[k,0] = 0
                    for n in range(0,np.size(data_select)):
                        mu_hat[k,0] = mu_hat[k,0] + (gamma_hat[n,k]*data_select[n,0])
                    mu_hat[k,0] = (1/N_k)*mu_hat[k,0]
                    var_hat[k,0] = 0
                    for n in range(0,np.size(data_select)):
                        var_hat[k,0] = var_hat[k,0] + (gamma_hat[n,k]*((data_select[n,0] - mu_hat[k,0])*(data_select[n,0] - mu_hat[k,0])))
                    var_hat[k,0] = (1/N_k)*var_hat[k,0]
                    pi_hat[k,0] = N_k/np.size(data_select)
                    if iteration == 0:
                        pi_hat_all = pi_hat
                    else:
                        pi_hat_all = np.hstack((pi_hat_all, pi_hat))
                likelihood = 0
                for n in range(0,np.size(data_select)):
                    likelihood = likelihood + np.log(np.sum(pi_hat[:,0]*gauss_fun(data_select[n,0], mu_hat[:,0], var_hat[:,0])))
                if iteration == 0:
                    likelihood_iteration = likelihood
                else:
                    likelihood_iteration = np.append(likelihood_iteration, likelihood)
                if iteration > 0:
                    likelihood_del = np.abs(likelihood_iteration[-1]) - np.abs(likelihood_iteration[-2])
                iteration = iteration + 1
            if epoch == 0:
                likelihood_epoch = likelihood_iteration[-1]
                mu_hat_epoch = mu_hat
                var_hat_epoch = var_hat
                pi_hat_epoch = pi_hat
            else:
                likelihood_epoch = np.append(likelihood_epoch, likelihood_iteration[-1])
                mu_hat_epoch = np.hstack((mu_hat_epoch, mu_hat))
                var_hat_epoch = np.hstack((var_hat_epoch, var_hat))
                pi_hat_epoch = np.hstack((pi_hat_epoch, pi_hat))
            epoch = epoch + 1
        best_likelihood_epoch_idx = np.argmax(likelihood_epoch)
        if K == 2:
            likelihood_K = likelihood_epoch[best_likelihood_epoch_idx]
            mu_hat_K = mu_hat_epoch[:,best_likelihood_epoch_idx]
            var_hat_K = var_hat_epoch[:,best_likelihood_epoch_idx]
            pi_hat_K = pi_hat_epoch[:,best_likelihood_epoch_idx]
        else:
            likelihood_K = np.append(likelihood_K, likelihood_epoch[best_likelihood_epoch_idx])
            mu_hat_K = np.append(mu_hat_K, mu_hat_epoch[:,best_likelihood_epoch_idx])
            var_hat_K = np.append(var_hat_K, var_hat_epoch[:,best_likelihood_epoch_idx])
            pi_hat_K = np.append(pi_hat_K, pi_hat_epoch[:,best_likelihood_epoch_idx])
        K = K + 1
    return likelihood_K, mu_hat_K, var_hat_K, pi_hat_K

likelihood_K, mu_hat_K, var_hat_K, pi_hat_K = em(max_K)
best_likelihood_K_idx = np.argmax(likelihood_K)
K = best_likelihood_K_idx + 2
print('Selecting Number of Gaussians =', K)
start_idx = int(np.sum(range(2,K)))
mu_hat = (mu_hat_K[start_idx:start_idx+K]).reshape(K,1)
var_hat = (var_hat_K[start_idx:start_idx+K]).reshape(K,1)
pi_hat = (pi_hat_K[start_idx:start_idx+K]).reshape(K,1)

pd = np.full((data_select.shape[0], K), float(0))
pd_mix = 0

for k in range(0,K):
    for n in range(0,data_select.shape[0]):
         x = gauss_fun(data_select[n,0], mu_hat[k,0], var_hat[k,0])
         pd[n,k] = x
    pd_mix = pd_mix + pi_hat[k,0]*pd[:,k]

plt.rcParams.update({'font.size': 20})

plt.figure(1)
plt.plot(data_select[:,0], pd_mix, 'r', label = 'pdf estimate')
plt.hist(data_select[:,0], density = True, label = 'True Data')
plt.legend()
plt.xlabel('Data')
plt.ylabel('PDF')
plt.title('True data compared to Estimated pdf, Number of Gaussians = %i' %K)
plt.rcParams['lines.linewidth'] = 3
plt.show

plt.figure(2)
plt.plot(range(2,max_K+1), likelihood_K)
plt.xlabel('Number of Gaussians')
plt.ylabel('Converged MLE')
plt.title('Maximum Likelihood w.r.t # Gaussians')
plt.show

