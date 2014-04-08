# Computes simple error for weighted classification
# y: original class labels
# y_prime: new class labels
# w: weights
classifier_error <- function(y, y_prime, w=vector()) {
    error <- 0
    n <- length(y)
    if(length(w) <= 0) w <- rep(1,n)
    for(i in 1:n) {
        if(y[i] != y_prime[i]) error <- error + w[i]
    }
    return(error / sum(w))
}


# takes in a data frame and randomly samples into K subsets
# the samples are marked by an integer in the 'folds' column representing
# which fold they belong to. Filtering can be accomplished by
# subset(.,select=...)
split_k_fold <- function(X, y, K=5) {
    folds <- sample(1:K, nrow(X), replace=TRUE)
    return(list(X=X, y=y, folds=folds)) # gives you a list with the 5 splits
}





# Weak Learner function (Decision Stump)
# 
# X: a dataframe containing training data points
# w: vector containing the weights for each training vector x
# y: vector containing class labels for each training vector x
# return: a list which contains the parameters specifying the resulting 
#   classifier, here a triplet (j, theta, m) specifying the decision stump
# 
train_decision_stump <- function(X, w, y) {
    # get dimensions
    d <- ncol(X)
    n <- nrow(X)
    
    best_score <- 0
    best_j <- 1
    best_theta <- 0

    for(j in 1:d) {
        # re-initialize score for each dimension
        score <- 0

        # sort indicies based on dimension so that we can find the
        # optimal split in linear time
        indicies <- order(X[,j])
        last_i <- indicies[1]
        # iterate over sorted indicies, and modify the score by the weighted
        # class values updating the model if the best score is surpassed
        # Achieves optimal split in O(nd)
        for(i in indicies[2:n]) {
            score <- score - 2*w[i]*y[i]
            if(X[i,j] != X[last_i,j] && abs(score) > abs(best_score)) {
                best_score <- score
                best_j <- j
                best_theta <- (X[i,j] + X[last_i,j]) / 2
            }
            last_i <- i
        }
    }

    return (c(best_j, best_theta, sign(best_score)))
}


# Applies a Weak learner to the data and gives class labels
# 
# X: the training data (rows are vectors of the data)
# pars: the result of a training function, here a triplet specifying decision
#   stumps (j, theta, m)
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

