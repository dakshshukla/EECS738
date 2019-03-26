# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 18:16:58 2019

@author: daksh
"""

import string
import numpy as np
import pickle


""" For testing only start"""
file_name = 'Shakesphere.txt'
num_lines = 100
num_states = 5
wordsPsentence = 10
""" For testing only end"""

# Input arguments:
#file_name = sys.argv[1]
#num_lines = int(sys.argv[2])
#lines = int(sys.argv[3])
#wordsPsentence = int(sys.argv[4])

with open ('vocab', 'rb') as fp:
    vocab = pickle.load(fp)

prob_init = np.load('prob_init_data.npy')
trans = np.load('trans_data.npy')
file_cons = open(file_name)

for ii in range(0,num_lines):
    line = file_cons.readline()
#split_words = (line.rstrip().lower())
#words = split_words.translate(str.maketrans('','', string.punctuation)).split()
#num_words = len(words)
##if num_words < num_states:
##    continue
#fwd = []
#p_now = np.zeros((num_states,0))
#f_prev = p_now
#for i in range(0,num_words):
#    for st in range(0,num_states):
#        if i == 0:
#            sum_prev = prob_init[st,0]
#        else:
#            sum_prev = sum(f_prev*trans[0:num_states,st].reshape((num_states,1)))
#
##            p_now[st] = trans[st,vocab.index(vocab[i])]*sum_prev
#        p_now[st] = sum_prev
#    fwd.append(p_now)
#    f_prev = p_now
#
#p_fwd = sum(p_now * trans[0:num_states,st].reshape((num_states,1)))
#p_fwd.append(p_fwd)

#line = file_cons.readline()
split_words = (line.rstrip().lower())
words = split_words.translate(str.maketrans('','', string.punctuation)).split()
num_words = len(words)
alpha = np.zeros((num_words,1))
beta = np.zeros((num_words,1))

for i in range(0,num_words):
    wrd = words[i]
    idx = vocab.index(wrd)
    alpha[i,0] = prob_init[idx,0]

for i in range(0,num_words):
    wrd = words[-i-1]
    idx = vocab.index(wrd)
    beta[i,0] = prob_init[idx,0]

for i in range(0,num_words):
    if i == 0:
        alpha[i,0] = alpha[i,0]*alpha[i,0]
    else:
        for ii in range(0,i):
            wrd_prev = words[ii]
            wrd_now = words[ii+1]
            idx_prev = vocab.index(wrd_prev)
            idx_now = vocab.index(wrd_now)
            alpha[i,0] = alpha[i,0] + alpha[i,0]*trans[idx_prev,idx_now]

for i in range(0,num_words):
    if i == 0:
        beta[i,0] = beta[i,0]*beta[i,0]
    else:
        for ii in range(0,i):
            wrd_last = words[-ii]
            wrd_2_last = words[-ii-1]
            idx_last = vocab.index(wrd_last)
            idx_2_last = vocab.index(wrd_2_last)
            beta[i,0] = beta[i,0] + beta[i,0]*trans[idx_last,idx_2_last]











