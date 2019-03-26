This is project number 2 for EECS 738. All files are written in Python 3.7

File 1: compute_trans_generate.py - This code finds the initial probabilities for the starting word of each sentence in the play,
and also finds the transition probability matrix, the elements of which are probabilities for each word transitions.

It can be run as follows:

python compute_trans_generate.py Shakesphere.txt 2000 5 10

It takes in 4 arguments: file name, number of lines to train from, number of sentences to generate, and words per sentence.
It functions as follows:

1. Taking the file name and the number of lines as inputs, the code first builds a vocabulary of unique words from the whole data
set of the set number of lines

2. After building the vocabulary, it computes the initial probabilities of the words that exist as the first words of the sentences in the data.
It also computes the transition probabilities for transitioning from one word to another, and saves all these values in a matrix.
In other words, it generates a probability transition matrix.

3. Then it generates text given the number of sentences and words to be generated per sentence.

4. Finally it saves the initial probability vector, probability transition matrix and the vocabulary list as python binary files.

This code's transition probability matrix and probability init vector are used as inputs for forward-backward algorithm,
generating text using Viterbi algorithm and also predicting text using Viterbi algorithm.

__________________________________________________________________________________________________________________

File 2: fwd_bwd.py - Computes marginal probabilities for the observed data using forward and backward steps,
and computes smoothed probabilities for sequential text.

It can be run as follows:

python fwd_bwd.py Shakesphere.txt 1000

It takes in 2 arguments: the input file and the line number for which the marginal probabilities are computed
for the sequency of words. It functions as follows:

1. Loads in binary data for initial probability matrix, probability transition matrix and vocabulary list.

2. Computes alpha and beta variables based in sequential data and transition probabilities

3. performs the smoothing step and computes the final marginal probabilities, and prints them for each word.


__________________________________________________________________________________________________________________

File 3: generate_text_viterbi.py - This code generates sentences based on user input and the sequential words
are predicted or chosen based on maximal probabilities - maxmimal probability route as in Viterbi algorithm.

It can be run as follows:

python generate_text_viterbi.py Shakesphere.txt 5 5

It takes in 2 arguments: the input file, number of sentences to genrate, and words per sentence to generate.
It functions as follows:

1. Loads in binary data for initial probability matrix, probability transition matrix and vocabulary list.

2. Randomly chooses an initial word based on initial probability vector.

3. Chooses the next word based on the previous word choosing the maximal transition probability from the matrix.
Continues for the "worde per sentence" input


__________________________________________________________________________________________________________________

File 4: predict_text_viterbi.py - This code suggests sequential words one at a time to a user.
It functions like a text completion algorithm. IMPORTANT: Choose words only from the vocabulary.

It can be run as follows:

python predict_text_viterbi.py Shakesphere.txt 5

It takes in 2 arguments: the input file and words per sentence to generate.
It functions as follows:

1. Loads in binary data for initial probability matrix, probability transition matrix and vocabulary list.

2. Asks user input for a number taht corresponds to the number of words to display from the vocabulary, for
the user to choose from.

3. It diplays the words from the vocabulary and shows intructions on how to type in
consecutive words by rejecting suggestions, or accept suggestions. The instructions displayed are:

"Type in a starting word and I will suggest a next word in a [box], type 'a' to accept or 'r' to reject,
if you reject the word then please type in another word from the vocabulary"

4. After receiving the first word from the user, it suggests a word form the vocabulary based on 
Viterbi algorithm (maximal probability).

5. The user can enter "a" to accept or "r" to reject. If "a" is entered it takes in the suggested word as
a part of sentence , otherwise the user has to input another word form the vocabulary.

6. The process continues until the "words per sentence" is reached.

7. It displays the final sentence in the end.




