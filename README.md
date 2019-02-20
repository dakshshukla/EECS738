# EECS738
This is Daksh Shukla's repository for EECS 738 course (Spring 2019) at The university of Kansas.

Project 1:
Some results of some selected features from particular datasets are shown as PNG images.
Example: "feature_1_winquality_like" is the result of EM GMM on column number 1 data from
winequality-red.csv file, and it shows the maximum likelihood with respect to number of Gaussians.
"feature_1_winequality_pdf"  is the result of EM GMM on column number 1 data from
winequality-red.csv file, and it shows the true data histogram and the estimated mixed
probability density function.

Some technical parts of the project were discussed with Mr. Ben Liu who helped with
understanding the fundamentals and formulate ideas to carry out the project.

File 1: EM_GMM.py - Main code for Expectation Maximization (EM) algorithm for Gaussian Mixture Models (GMM).
The code is based on section 9.2.2 of Pattern Recognition and Machine Learning by Christopher M. Bishop.

1. Running the code: The code takes in 3 arguments
	a) <filename>.csv
	b) Column number of the csv file that is the feature
	c) Maximum number of Gaussians (K) to try out (max_K)
	d) Maximum epochs or iterations for different initial guesses of means for a particular K (max_epoch)
Example: 'python3 winequality-red.csv 1 5 50'
Caution: for large datasets higher numbers of max_epoch (>50) and higher numbers for max_K (>5) can cause slow processing

2. The code computes the means and variances for the respective number of Gaussians (denoted by capital K),
and it does this max_epoch times while choosing different means as the starting point in the EM algorithm

3. Then it computes the log likelihood for this particular mixture for max_epoch different initial conditions and
selects the epoch with best (highest) maximum likelihood.

4. It increments the K value (number of Gaussians) up to the maximum K input and re-computes means and variances and
the corresponding log-likelihood (goes back to step 2)

5. After going through all K's, it compares and selects the best Gaussian mixture K value based on best (highest)
maximum likelihood value

6. Finally it computes the mixture pdf based on these selected parameters and plots and compares the
estimated pdf with the true data

-------------------------------------------------------------------------------------------------------------------------------------------------------

File 2: em_esl8_5.py - Code for practicing and understanding the EM algorithm for a Gaussian mixture 2 distributions.
This code is based on section 8.5 of book The Elements of Statistical Learning by Trevor Hastie, Robert Tibshirani and Jerome Friedman.

1. Running the code: The code does not take any arguments and runs simply as 'python3 em_esl8_5.py'
It already has the data vector from the book's section hard-coded in a numpyy array.

2. It computes means and variances for a mixture of 2 Gaussians and iterates 300 times through selection of
different initial guesses of means.

3. It saves all these means, variances and also the computed log-likelihood for every iteration

4. Then it selects the parameters for the best or highest log-likelihood value

5. Finally it computes the mixture pdf and compares it with the true data on a plot.