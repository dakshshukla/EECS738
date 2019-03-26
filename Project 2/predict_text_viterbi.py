# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 19:25:12 2019

@author: daksh
"""

# Command for runnning via IPython console in Spyder:
# runfile('predict_text_viterbi.py', args = 'Shakesphere.txt 5')
# arguments = file name, and words per sentence to generate

import numpy as np
import pickle
import os
import sys

""" For testing only start"""
#file_name = 'Shakesphere.txt'
#wordsPsentence = 5
""" For testing only end"""

# Input arguments:
file_name = sys.argv[1]
wordsPsentence = int(sys.argv[2])

with open ('vocab', 'rb') as fp:
    vocab = pickle.load(fp)

prob_init = np.load('prob_init_data.npy')
trans = np.load('trans_data.npy')
file_cons = open(file_name)

num_words_disp = int(input('Enter number of words to display from the vocabulary: \n'))
vocab_len = len(vocab)

def print_words():
    print('Choose from the following words: \n')
    for i in range(0,num_words_disp):
        idx_wrd = np.asscalar(np.random.choice(vocab_len, 1, prob_init.tolist()))
        wrd = vocab[idx_wrd]
        print(wrd, end =" ")
    print('\n')
    print('Type in a starting word and I will suggest a next word in a [box], type \'a\' to accept or \'r\' to reject, if you reject the word then please type in another word from the vocabulary \n')

def generate_text(wordsPsentence,vocab):
    print_words()
    wrd_vocab = input()
    os.system('cls')
    print_words()
    print(wrd_vocab, end = ' ')
    sentence = [wrd_vocab]
    sent_final = [wrd_vocab]
    for ii in range(0,wordsPsentence):
        idx_wrd = vocab.index(wrd_vocab)
        state_vec = trans[:,idx_wrd]
        idx_next = np.argmax(state_vec)
        wrd_vocab = vocab[idx_next]
        sentence.append('['+wrd_vocab+']')
        print('['+wrd_vocab+']', end = ' ')
        accept_reject = input()
        if accept_reject == 'a':
            os.system('cls')
            print_words()
            print(*sentence, end = ' ')
            print(wrd_vocab, end = ' ')
            sentence.append(wrd_vocab)
            sent_final.append(wrd_vocab)
        elif accept_reject == 'r':
            os.system('cls')
            print_words()
            print(*sentence, end = ' ')
            wrd_vocab = input()
            os.system('cls')
            print_words()
            print(*sentence, end = ' ')
            print(wrd_vocab, end = ' ')
            sentence.append(wrd_vocab)
            sent_final.append(wrd_vocab)
    print('\n')
    print('Final Sentence: \n')
    print(*sent_final)

generate_text(wordsPsentence,vocab)

