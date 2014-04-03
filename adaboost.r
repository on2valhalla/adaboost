# Homework problem 1
# Implement adaboost algorithm

# function calls of form
# pars <- train(X, w, y)
# label <- classify(X, pars)
# c_hat <- agg_class(X, alpha, allPars)


# Weak Learner function (Decision Stump)
# 
# X: a matrix, columns are training vectors x1..xn
# w: vector containing the weights for each training vector x
# y: vector containing class labels for each training vector x
# return: a list which contains the parameters specifying the resulting 
# 	classifier, here a triplet (j, theta, m) specifying the decision stump
# 
train_decision_stump <- function(X, w, y) {

	return (j, theta, m)
}


# Applies a Weak learner to the data and gives class labels
# 
# X: the training data (columns are vectors of the data)
# pars: the result of a training function, here a triplet specifying decision
# 	stumps (j, theta, m)
classify_decision_stump <- function(X, pars) {
	n <- length(X[,1])
	j <- pars[1]
	theta <- pars[2]
	m <- pars[3]
	# initialize class labels
	y_prime <- vector(mode='numeric',length=n)

	for(i in 1:n) {
		y_prime[i] <- ifelse(X[j,i] > theta, m, -m)
	}
	return y_prime
}


# X: training data (columns are vectors of points)
# alpha: denotes the vector of voting weights
# allPars: contains the parameters of all the weak learners
agg_class <- function(X, alpha, allPars) {

}