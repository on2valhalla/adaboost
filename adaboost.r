# Homework problem 1
# Implement adaboost algorithm

# function calls of form
# pars <- train(X, w, y)
# label <- classify(X, pars)
# c_hat <- agg_class(X, alpha, allPars)


# X: a matrix, columns are training vectors x1..xn
# w: vector containing the weights for each training vector x
# y: vector containing class labels for each training vector x
# return: a list which contains the parameters specifying the resulting 
# 	classifier, here a triplet (j, theta, m) specifying the decision stump
# 
train <- function(X, w, y) {

	return (j, theta, m)
}

# X: the training data (columns are vectors of the data)
# pars: the result of a training function, here a triplet specifying decision
# 	stumps
label <- function(X, pars) {

}


# X: training data (columns are vectors of points)
# alpha: denotes the vector of voting weights
# allPars: contains the parameters of all the weak learners
agg_class <- function(X, alpha, allPars) {

}