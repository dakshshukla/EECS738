# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 18:16:58 2019

@author: daksh
"""

# Command for runnning via IPython console in Spyder:
# runfile('fwd_bwd.py', args = 'Shakesphere.txt 1000')
# arguments = file name, line number (for observations)

import string
import numpy as np
import pickle
import sys

""" For testing only start"""
#file_name = 'Shakesphere.txt'
#line_num = 1000
""" For testing only end"""

# Input arguments:
file_name = sys.argv[1]
line_num = int(sys.argv[2])

with open ('vocab', 'rb') as fp:
    vocab = pickle.load(fp)

prob_init = np.load('prob_init_data.npy')
trans = np.load('trans_data.npy')
file_cons = open(file_name)

for ii in range(0,line_num):
    line = file_cons.readline()

split_words = (line.rstrip().lower())
words = split_words.translate(str.maketrans('','', string.punctuation)).split()
num_words = len(words)
alpha = np.zeros((num_words,1))
beta = np.ones((num_words,1))
beta = beta/sum(beta)

for i in range(0,num_words):
    wrd = words[i]
    idx = vocab.index(wrd)
    alpha[i,0] = prob_init[idx,0]
    
    if i == 0:
        alpha[i,0] = alpha[i,0]
    else:
        for ii in range(0,i):
            wrd_prev = words[ii]
            wrd_now = words[ii+1]
            idx_prev = vocab.index(wrd_prev)
            idx_now = vocab.index(wrd_now)
            if ii == 0:
                alpha[i,0] = alpha[i,0]*trans[idx_prev,idx_now]
            else:
                alpha[i,0] = alpha[i,0] + alpha[i,0]*trans[idx_prev,idx_now]

            wrd_last = words[-ii]
            wrd_2_last = words[-ii-1]
            idx_last = vocab.index(wrd_last)
            idx_2_last = vocab.index(wrd_2_last)
            if ii == 0:
                beta[-i,0] = beta[-i,0] + beta[-i,0]*trans[idx_last,idx_2_last]
            else:
                beta[-i,0] = beta[-i,0] + beta[-i,0]*trans[idx_last,idx_2_last]

p_post = alpha*beta
p_post = p_post/sum(p_post)

print('\n')
print('Marginal Posterior Probabilities for line number:', line_num , '\n')
for i in range(0,num_words):
    print(words[i], p_post[i], '\n')

