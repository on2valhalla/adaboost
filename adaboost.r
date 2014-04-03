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
		y_prime[i] <- ifelse(X[i,j] > theta, m, -m)
	}
	return(y_prime)
}


# X: training data (columns are vectors of points)
# alpha: denotes the vector of voting weights
# allPars: contains the parameters of all the weak learners
agg_class <- function(X, alpha, allPars) {

}

adaboost <- function(X, y, train, classify) {

}