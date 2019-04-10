## Problem

Obtain the activities of daily life dataset from the UC Irvine machine learning website (https://archive.ics.uci.edu/ml/datasets/Dataset+for+ADL+Recognition+with+Wrist-worn+Accelerometer
data provided by Barbara Bruno, Fulvio Mastrogiovanni and Antonio Sgorbissa).


(a) Build a classifier that classifies sequences into one of the 14 activities provided. To make features, you should vector quantize, then use a histogram of cluster centers (as described in the subsection; this gives a pretty explicit set of steps to follow). You will find it helpful to use hierarchical k-means to vector quantize. You may use whatever multi-class classifier you wish, though I’d start with R’s decision forest, because it’s easy to
use and effective. You should report 

(a.1) the total error rate and 

(a.2) the class confusion matrix of your classifier.


(b) Now see if you can improve your classifier by 

(b.1) modifying the number of cluster centers in your hierarchical k-means 

(b.2) modifying the size of the fixed length samples that you use
