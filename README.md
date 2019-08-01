# LinearRegressionExample

This is an example implementation of Linear Regression done on python, I provide my code and the dataset used.

Here is the prompt of the assignment:

1. Start the experiment by creating 3 additional training files from the train-1000-100.csv by taking
the first 50, 100, and 150 instances respectively. Call them: train-50(1000)-100.csv, train-100(1000)-
100.csv, train-150(1000)-100.csv. The corresponding test file for these dataset would be test-1000-
100.csv and no modifcation is needed.

2. Implement L2 regularized linear regression algorithm with λ ranging from 0 to 150 (integers only). For
each of the 6 dataset, plot both the training set MSE and the test set MSE as a function of  λ (x-axis)
in one graph.

3. From the plots in question 2, we can tell which value of λ is best for each dataset once we know the
test data and its labels. This is not realistic in real world applications. In this question, you will use
cross validation (CV) to set the value for λ. Implement the 10-fold CV technique discussed in class
(pseudo code given in Appendix A) to select the best λ value based on the training data only and
answer the following questions:

(a) Using the CV technique, what is the best choice of λ and the corresponding test set MSE for each
of the six datasets?
(b) How do the values for λ and MSE obtained from CV compare to the choice of λ and MSE in
question 2(a)?
(c) What are the drawbacks of CV?
(d) What are the factors affecting the performance of CV?

4. Fix λ = 1, 25, 150. For each of these values, plot a learning curve for the algorithm using the dataset
1000-100.csv.
