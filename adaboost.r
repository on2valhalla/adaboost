# Homework problem 1
# Implement adaboost algorithm

# function calls of form
# pars <- train(X, w, y)
# label <- classify(X, pars)
# c_hat <- agg_class(X, alpha, allPars)


# Weak Learner function (Decision Stump)
# 
# X: a matrix, rows are training vectors x1..xn
# w: vector containing the weights for each training vector x
# y: vector containing class labels for each training vector x
# return: a list which contains the parameters specifying the resulting 
# 	classifier, here a triplet (j, theta, m) specifying the decision stump
# 
train_decision_stump <- function(X, w, y) {
	# get dimensions
	d <- length(X[1,])
	n <- length(X[,1])

	init_score <- drop(w %*% y)
	best_score <- init_score
	best_j <- 1
	best_theta <- 0

	for(j in 1:d) {
		# sort based on dimension so that we can find the
		# optimal split in linear time
		# ordered_X <- X[,order(X[j,])]

		# re-initialize score for each dimension
		score <- init_score

		# sort indicies based on dimension so that we can find the
		# optimal split in linear time
		indicies <- order(X[,j])
		last_i <- indicies[1]
		# iterate over sorted indicies, and modify the score by the weighted
		# class values updating the model if the best score is surpassed
		# Achieves optimal split in O(nd)
		for(i in indicies[2:n]) {
			score <- score - w[i]*y[i]
			if(X[i,j] != X[last_i,j] && score > best_score) {
				best_score <- score
				best_j <- j
				best_theta <- (X[i,j] - X[last_i,j]) / 2
			}
		}
	}

	return (c(best_j, best_theta, sign(best_score)))
}


# Applies a Weak learner to the data and gives class labels
# 
# X: the training data (rows are vectors of the data)
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
		y_prime[i] <- ifelse(X[i,j] > theta, m, -m)
	}
	return(y_prime)
}


# X: training data (rows are vectors of points)
# voting_weights: denotes the vector of voting weights
# classifiers: contains the parameters of all the weak learners
aggregate_classifiers <- function(X, voting_weights, classifiers) {

}

adaboost <- function(X, y, train, classify, B) {
	n <- length(X[,1])
	# initialize weights to even distribution
	w <- rep(1/n,n)
	# initialize returns, voting_weights == alphas
	voting_weights <- rep(0,B)
	# parameters of classifiers
	classifiers <- matrix(nrow=B,ncol=3)

	for(b in 1:B) {
		# train a weak learner on the weighted data
		classifiers[b] <- train_decision_stump(X, w, y)
		# classify data with weak learner
		y_prime <- classify_decision_stump(X, classifiers[b])
		# compute the error of the weak learner
		error <- classifier_error(y, y_prime, w)
		# compute voting weights
		voting_weights[b] <- log((1-error)/error)
		# recompute weights
		for(i in 1:n) {
			if(y[i] != y_prime[i]) w[i] <- w[i] * exp(voting_weights[b])
		}
	}

	# return weights and parameters of weak learners
	return(list(voting_weights=voting_weights,classifiers=classifiers))
}

# Computes simple error for weighted classification
# y: original class labels
# y_prime: new class labels
# w: weights
classifier_error <- function(y, y_prime, w) {
	error <- 0
	for(i in 1:length(y)) {
		if(y[i] != y_prime[i]) error <- error + w[i]
	}
	return(error / sum(w))
}

