# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 12:23:27 2019

@author: daksh
"""
# Command for runnning via IPython console in Spyder:
# runfile('computeTrans_generate.py', args = 'Shakesphere.txt 2000 5 10')
# arguments = file name, number of lines to train from,
#number of sentences to generate, and words per sentence

import string
import numpy as np
import sys
from tempfile import TemporaryFile
import os
import pickle

dir_path = os.path.dirname(os.path.realpath(__file__))
#import random

#random.seed(1648)
#np.random.seed(1236)

""" For testing only start"""
#file_name = 'Shakesphere.txt'
#num_lines = 2000
#lines = 5
#wordsPsentence = 10
""" For testing only end"""

# Input arguments:
file_name = sys.argv[1]
num_lines = int(sys.argv[2])
lines = int(sys.argv[3])
wordsPsentence = int(sys.argv[4])

def build_vocab(file_name,num_lines):
    file_cons = open(file_name)
    vocab = []
    for ii in range(0,num_lines):
        line = file_cons.readline()
        split_words = (line.rstrip().lower())
        words = split_words.translate(str.maketrans('','', string.punctuation)).split()
        num_words = len(words)
        
        for i in range(num_words):
            wrd = words[i]
            if wrd in vocab:
                continue
            else:
                vocab.append(wrd)
    return vocab

def getinit_trans(file_name,num_lines,vocab):
    file_cons = open(file_name)
    vocab_len = len(vocab)
    count_first_word = np.zeros((vocab_len,1))
    count_trans = np.zeros((vocab_len,vocab_len))
    
    for ii in range(0,num_lines):
        line = file_cons.readline()
        split_words = (line.rstrip().lower())
        words = split_words.translate(str.maketrans('','', string.punctuation)).split()
        num_words = len(words)
        for i in range(0,num_words):
            if i == 0:
                idx_vocab = vocab.index(words[i])
                count_first_word[idx_vocab,0] = count_first_word[idx_vocab,0] + 1
            else:
                idx_prev = vocab.index(words[i-1])
                idx_now = vocab.index(words[i])
                count_trans[idx_prev,idx_now] = count_trans[idx_prev,idx_now] + 1
                
    prob_init = count_first_word/sum(count_first_word)
    trans = np.zeros((vocab_len,vocab_len))
    
    for i in range(0,vocab_len):
        if sum(count_trans[i,:]) != 0:
            trans[i,:] = count_trans[i,:]/sum(count_trans[i,:])
    
    return prob_init, trans

def generate_text(lines,wordsPsentence,vocab):
    vocab_len = len(vocab)
    for ii in range(0,lines):
        idx_wrd = np.asscalar(np.random.choice(vocab_len, 1, prob_init.tolist()))
        wrd_vocab = vocab[idx_wrd]
        for i in range(0,wordsPsentence):
            state_vec = trans[:,idx_wrd]
            idx_next = np.asscalar(np.random.choice(len(state_vec), 1, state_vec.tolist()))
            #idx_next = np.argmax(state_vec)
            print(wrd_vocab, end = " ")
            wrd_vocab = vocab[idx_next]
            idx_wrd = vocab.index(wrd_vocab)
        print('\n')

vocab = build_vocab(file_name,num_lines)
prob_init, trans = getinit_trans(file_name,num_lines,vocab)
generate_text(lines, wordsPsentence,vocab)

prob_init_data = TemporaryFile()
np.save(os.path.join(dir_path, 'prob_init_data'), prob_init)
trans_data = TemporaryFile()
np.save(os.path.join(dir_path, 'trans_data'), trans)

with open('vocab', 'wb') as fp:
    pickle.dump(vocab, fp)
