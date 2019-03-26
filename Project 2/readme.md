This is project number 2 for EECS 738.

File 1: computeTrans_generate.py is written in python 3, and can be run as follows:

python Shakesphere.txt 2000 5 10

It takes in 4 arguments: file name, number of lines to train from, number of sentences to generate, and words per sentence

The code is based on training or learning probability values for transitioning from one state (word) to another. It functions as follows:

1. Taking the file name and the number of lines as inputs, the code first builds a vocabulary of unique words from the whole data
set of the set number of lines

2. After building the vocabulary, it computes the initial probabilities of the words that exist as the first words of the sentences in the data.
It also computes the transition probabilities for transitioning from one word to another, and saves all these values in a matrix.
in other words, it generates a probability transition matrix.

3. Then it generates text given the number of sentences and words to be generated per sentence.

This code's transition probability matrix and probability init vector are used as inputs for forward-backward algorithm


