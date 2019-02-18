# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 18:29:45 2019

@author: daksh

This code is developed based on section 8.5 of the ESL book to help
understand the EM algorithm use for modelling a two component 
mixture of Gaussians
"""

import numpy as np
import statistics as st
import matplotlib.pyplot as plt

data = np.asarray([[-0.39, 0.12, 0.94, 1.67, 1.76, 2.44, 3.72, 4.28, 4.92, 5.53, 0.06, 0.48, 1.01, 1.68, 1.80, 3.25, 4.12, 4.60, 5.28, 6.22]])
data = np.sort(data)

stop_eps = 1e-10
iteration = 0

def gauss_fun(x, mu, var):
    pd = (1/(np.sqrt(2*np.pi*var)))*np.exp(-1*np.square(x-mu)/(2*var))
    return pd

while iteration < 300:
    idx1 = np.random.choice(data.shape[1], 1, replace=False)
    idx2 = np.random.choice(data.shape[1], 1, replace=False)
    mu1 = data[0,idx1]
    mu2 = data[0,idx2]
    var1 = np.sum(np.square(data[0,:] - np.mean(data[0,:])))/20
    var2 = np.sum(np.square(data[0,:] - np.mean(data[0,:])))/20
    pi_hat = 0.5
    pi_hat_ar = np.array([])
    likeli_ar = np.array([])
    pi_hat_del = 1
    ii = 0
    while(pi_hat_del > stop_eps):
        gamma_hat_ar = np.array([])
        for i in range(0,20):
            # Expectation Step:
            gamma_hat = (pi_hat*gauss_fun(data[0,i], mu2, var2))/((1-pi_hat)*gauss_fun(data[0,i], mu1, var1) + pi_hat*gauss_fun(data[0,i], mu2, var2))
            gamma_hat_ar = np.append(gamma_hat_ar, gamma_hat)
        # Maximization Step:
        mu1 = np.sum(np.multiply(1-gamma_hat_ar, data[0,:]))/np.sum(1-gamma_hat_ar)
        mu2 = np.sum(np.multiply(gamma_hat_ar, data[0,:]))/np.sum(gamma_hat_ar)
        
        var1 = np.sum(np.multiply(1-gamma_hat_ar, np.square(data[0,:]-mu1)))/np.sum(1-gamma_hat_ar)
        var2 = np.sum(np.multiply(gamma_hat_ar, np.square(data[0,:]-mu2)))/np.sum(1-gamma_hat_ar)
        
        pi_hat = np.sum(gamma_hat_ar/20)
        pi_hat_ar = np.append(pi_hat_ar, pi_hat)
        if ii > 0:
            pi_hat_del = pi_hat_ar[-1] - pi_hat_ar[-2]
        likeli = np.sum(np.log((1-pi_hat)*gauss_fun(data[0,:], mu1, var1) + pi_hat*gauss_fun(data[0,:], mu2, var2)))
        likeli_ar = np.append(likeli_ar, likeli)
        ii = ii + 1
    if iteration == 0:
        likeli_iter = likeli_ar[-1]
        mu1_iter = mu1
        mu2_iter = mu2
        var1_iter = var1
        var2_iter = var2
    else:
        likeli_iter = np.append(likeli_iter, likeli_ar[-1])
        mu1_iter = np.append(mu1_iter, mu1)
        mu2_iter = np.append(mu2_iter, mu2)
        var1_iter = np.append(var1_iter, var1)
        var2_iter = np.append(var2_iter, var2)
    best_likeli_idx = np.argmax(likeli_iter)
    iteration = iteration + 1

pd_arr = np.array([])

for i in range(0,20):
    pd1 = gauss_fun(data[0,i], mu1_iter[best_likeli_idx], var1_iter[best_likeli_idx])
    pd2 = gauss_fun(data[0,i], mu2_iter[best_likeli_idx], var2_iter[best_likeli_idx])
    pd = (1-pi_hat)*pd1 + pi_hat*pd2
    pd_arr = np.append(pd_arr, pd)

plt.plot(data[0,:], pd_arr, 'r', label = 'pdf estimate')
plt.hist(data[0,:], density=True, label = 'True Data')
plt.legend()
plt.show

    

