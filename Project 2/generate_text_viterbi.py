# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:52:31 2019

@author: daksh
"""

# Command for runnning via IPython console in Spyder:
# runfile('generate_text_viterbi.py', args = 'Shakesphere.txt 5 5')
# arguments = file name, number of sentences to generate, and words per sentence

import numpy as np
import pickle
import sys

""" For testing only start"""
#file_name = 'Shakesphere.txt'
#lines = 5
#wordsPsentence = 5
""" For testing only end"""

# Input arguments:
file_name = sys.argv[1]
lines = int(sys.argv[2])
wordsPsentence = int(sys.argv[3])

with open ('vocab', 'rb') as fp:
    vocab = pickle.load(fp)

prob_init = np.load('prob_init_data.npy')
trans = np.load('trans_data.npy')
file_cons = open(file_name)

def generate_text(lines,wordsPsentence,vocab):
    vocab_len = len(vocab)
    for ii in range(0,lines):
        idx_wrd = np.asscalar(np.random.choice(vocab_len, 1, prob_init.tolist()))
        wrd_vocab = vocab[idx_wrd]
        for i in range(0,wordsPsentence):
            state_vec = trans[:,idx_wrd]
            #idx_next = np.asscalar(np.random.choice(len(state_vec), 1, state_vec.tolist()))
            idx_next = np.argmax(state_vec)
            print(wrd_vocab, end = " ")
            wrd_vocab = vocab[idx_next]
            idx_wrd = vocab.index(wrd_vocab)
        print('\n')

generate_text(lines, wordsPsentence,vocab)

