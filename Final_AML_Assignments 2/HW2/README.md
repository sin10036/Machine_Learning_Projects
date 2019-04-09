## Problem 1 (support vector machine data using stochastic gradient descent)


The UC Irvine machine learning data repository hosts a collection of data on adult income, donated by Ronny Kohavi and Barry Becker. You can find this data at https://archive.ics.uci.edu/ml/datasets/Adult For each record, there is a set of continuous attributes, and a class "less than 50K" or "greater than 50K". We will provide you with 43958 examples with known class labels, and 4884 examples without class labels, to be found at https://www.kaggle.com/t/f3528db914934de29c24d30cf792dea3. Split it randomly into 10% validation and 90% training data.

Write a program to train a support vector machine on this data using stochastic gradient descent. You should not use a package to train the classifier (that's the point), but your own code. You should ignore the id number, and use the continuous variables as a feature vector. You should scale these variables so that each has unit variance, and you should subtract the mean so that each has zero mean. You should search for an appropriate value of the regularization constant, trying at least the values [1e-3, 1e-2, 1e-1, 1]. Use the validation set for this search. You should use at least 50 epochs of at least 300 steps each. In each epoch, you should separate out 50 training examples at random for evaluation (call this the set held out for the epoch). You should compute the accuracy of the current classifier on the set held out for the epoch every 30 steps. You should produce:

A. A screenshot of your leaderboard accuracy.

B. A plot of the accuracy every 30 steps, for each value of the regularization constant.

C. A plot of the magnitude of the coefficient vector every 30 steps, for each value of the regularization constant.

D. Your estimate of the best value of the regularization constant, together with a brief description of why you believe that is a good value.

E. What was your choice for the learning rate and why did you choose it ?
