# Homework problem 1
# Implement adaboost algorithm

# function calls of form
# pars <- train(X, w, y)
# label <- classify(X, pars)
# c_hat <- agg_class(X, alpha, allPars)

# X: training data, row is a data point
# B: Number of classifiers
run_adaboost <- function(X, y, B) {
	data <- cbind(data.frame(y),data.frame(X))
	c <- adaboost(data,train_decision_stump,classify_decision_stump,B)
	y_prime <- aggregate_weak_classifiers(X, c$voting_weights, c$classifiers)
	error <- classifier_error(y,y_prime,rep(1,length(y)))
	return(list(c=c,y_prime=y_prime,error=error))
}


# Weak Learner function (Decision Stump)
# 
# X: a dataframe containing training data points
# w: vector containing the weights for each training vector x
# y: vector containing class labels for each training vector x
# return: a list which contains the parameters specifying the resulting 
# 	classifier, here a triplet (j, theta, m) specifying the decision stump
# 
train_decision_stump <- function(X, w, y) {
	# get dimensions
	d <- ncol(X)
	n <- nrow(X)

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
	n <- nrow(X)
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


# Implementation of iterative adaboost algorithm
# X: training data (row is a data point)
# y: training classes
# train: training method for weak learner
# classify: classification method for weak learner
# B: number of weak learners to train
adaboost <- function(data, train, classify, B) {
    X <- subset(data,select=-y)
    y <- subset(data,select=y)[,1]
    
	n <- nrow(X)
	# initialize weights to even distribution
	w <- rep(1/n,n)
	# initialize returns, voting_weights == alphas
	voting_weights <- rep(0,B)
	# parameters of classifiers
	classifiers <- matrix(nrow=B,ncol=3)

	for(b in 1:B) {
	# train a weak learner on the weighted data
	classifiers[b,] <- train_decision_stump(X, w, y)
	# classify data with weak learner
	y_prime <- classify_decision_stump(X, classifiers[b,])
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


# X: training data (rows are vectors of points)
# voting_weights: denotes the vector of voting weights
# classifiers: contains the parameters of all the weak learners
# return: classification labels for data
aggregate_weak_classifiers <- function(X, voting_weights, classifiers) {
	n <- nrow(X)
	B <- length(classifiers[,1])
	y_prime <- vector(mode='numeric', length=n)
	y_votes <- matrix(nrow=B, ncol=n)

	for(b in 1:B) {
		y_votes[b,] <- classify_decision_stump(X,classifiers[b,])
	}
	for(i in 1:n) {
		weighted_class <- 0
		for(b in 1:B) {
			weighted_class <- weighted_class + voting_weights[b] * y_votes[b,i]
		}
		y_prime[i] <- sign(weighted_class)
	}
	return(y_prime)
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


# takes in a data frame and randomly samples into K subsets
# if you give this a matrix, it will split correctly, but return vectors
# as the list components instead of matrices
split_k_fold <- function(X, K=5) {
	indicies <- sample(1:K, nrow(X), replace=TRUE)
	return(split(x,indicies)) # gives you a list with the 5 splits
}
