# Spam-detector

This is my personal for fun project, where I wanted to use ML methods to detect spam.
The email database used is from: https://www2.aueb.gr/users/ion/data/enron-spam/

The SVM using 1000 most frequent 1-grams and 2-grams and enron 1-6 reaches up to 97% accuracy on the test set.
This is honestly more than I expected, but could probably be further improved by processing the input data more.

I very briefly tested the SVM on some of my emails and it performance drops, as expected. It mostly struggles with detecting a lot of false spam.
