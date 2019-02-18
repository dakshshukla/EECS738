# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 21:14:02 2019

@author: daksh
"""

# Command for runnning via IPython console in Spyder:
# runfile('EM_GMM.py', args = 'winequality-red.csv 1 5')
# runfile('EM_GMM.py', args = 'iris.csv 1 5')

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
#feature_num = 4
#max_K = 3
""" For testing only end"""

fname = sys.argv[1]
feature_num = int(sys.argv[2])
max_K = int(sys.argv[3])

fdata = pd.read_csv(fname)
data_col = fdata.get_values()
data_select = np.reshape(data_col[:,feature_num-1], (np.size(data_col,0),1))
data_select = np.sort(data_select, axis = 0)

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
                likelihood_all = likelihood
            else:
                likelihood_all = np.append(likelihood_all, likelihood)
            if iteration > 0:
                likelihood_del = np.abs(likelihood_all[-1]) - np.abs(likelihood_all[-2])
            iteration = iteration + 1
        if K == 2:
            likelihood_K = likelihood_all[-1]
        else:
            likelihood_K = np.append(likelihood_K, likelihood_all[-1])
        K = K + 1
    return likelihood_K, mu_hat, var_hat, pi_hat

likelihood_K, mu_hat, var_hat, pi_hat = em(max_K)
best_likeli_idx = np.argmax(likelihood_K)
K = best_likeli_idx + 2
print('Selecting Number of Gaussians =', K)
likelihood_K_sel, mu_hat, var_hat, pi_hat = em(K)

pd = np.full((data_select.shape[0], K-1), float(0))
pd_mix = 0

for k in range(0,K-1):
    for n in range(0,data_select.shape[0]):
         x = gauss_fun(data_select[n,0], mu_hat[k,0], var_hat[k,0])
         pd[n,k] = x
    pd_mix = pd_mix + pi_hat[k,0]*pd[:,k]

weights = np.ones_like(data_select[:,0])/data_select.shape[0]

plt.figure(1)
plt.plot(data_select[:,0], pd_mix, 'r', label = 'pdf estimate')
plt.hist(data_select[:,0], density = True, label = 'True Data')
plt.legend()
plt.title('Number of Gaussians = %i' %K)
plt.show

plt.figure(2)
plt.plot(range(2,max_K+1), likelihood_K)
plt.xlabel('Number of Gaussians')
plt.ylabel('Converged MLE')
plt.title('Maximum Likelihood w.r.t # Gaussians')
plt.show

