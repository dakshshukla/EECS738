mnist.py file: Run as python mnist.py

Code for single hidden layer ANN training with momenum, trains very fast

- Loads mnist data and its labels (the data is saved as npy file and was first downloaded from Scikit.learn)
- Divides data into training ans testing samples
- Asks the user if a new training needs to be done or if pretrained parameters can be used
- Asks the user for number of epochs for training (use around 10 epochs for good accuracy)
- Evaluates the trained model or with loaded parameters
- Asks the user what number picture shold be tested
- Shows the number picture and the predicted number with its accuracy percentage
________________________________________________________________________________________________________________________________________

Results folder contains some results from mnist_2 code.
________________________________________________________________________________________________________________________________________

Practice codes:
________________________________________________________________________________________________________________________________________

ann file: Basic neural network model for a simple regression problem

The code is based on the following blog post: https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
It trains the model for 2000 epochs and prints out true and predicted values.
________________________________________________________________________________________________________________________________________

mnist_1 file: Takes in "number of training epochs" as input argument, run as: "python mnist_1.py 500"

The code based on the following blog post: https://jonathanweisberg.org/post/A%20Neural%20Network%20from%20Scratch%20-%20Part%201/
The ANN is single layer (logistic regression) with sigmoid activations. It predicts the class of number "0" only after training.
After training it evaluates the model for all test cases and plots the predicted and true classes as "1" or "0"
with percentage accuracy. "1" class means that it is predicting "0" number.
________________________________________________________________________________________________________________________________________

mnist_2 file: no arguments run as: python mnist_2.py

The ANN model uses a single hidden layer with sigmoid activations and 64 nodes and the output layer is softmax. The code
is based of the following blog post: https://jonathanweisberg.org/post/A%20Neural%20Network%20from%20Scratch%20-%20Part%201/

- Loads mnist data and its labels (the data is saved as npy file and was first downloaded from Scikit.learn)
- Divides data into training ans testing samples
- Asks the user if a new training needs to be done or if pretrained parameters can be used
- Asks the user for number of epochs for training (use around 2000 epochs for good accuracy)
- Evaluates the trained model or with loaded parameters
- Asks the user what number picture shold be tested
- Shows the number picture and the predicted number with its accuracy percentage

