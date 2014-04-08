# Homework problem 1
# Implement adaboost algorithm

# function calls of form
# pars <- train(X, w, y)
# label <- classify(X, pars)
# c_hat <- agg_class(X, alpha, allPars)


library(reshape,plyr,ppls)


# Reads in data files and runs everything
run_adaboost <-function() {
    source('utils.r')
    X <- data.frame(read.table('uspsdata.txt'))
    y <- scan('uspscl.txt')

    results <- run_basic_adaboost(X,y,10)
    kfold_results <- run_kfold_adaboost(X,y,100,5)


    plot(lowess(results$errors$agg_train), type='l')
    plot(lowess(results$errors$agg_test), type='l')
    matplot(colMeans(kfold_results$errors$train), type='l',
        xlab='Adaboost Iteration',ylab='Aggregated Mean Training Error')
    matplot(colMeans(kfold_results$errors$test), type='l',
        xlab='Adaboost Iteration',ylab='Aggregated Mean Test Error')
}

# X: training data, row is a data point
# B: Number of classifiers
run_basic_adaboost <- function(X, y, B) {
    # idxs <- sample(1:nrow(X), nrow(X) * 0.8, replace=FALSE)
    idxs <- 1:floor(.8*nrow(X))
    test_data <- list(X=X[-idxs,], y=y[-idxs])
    tr_data <- list(X=X[idxs,], y=y[idxs])

    # c <- adaboost(tr_data, test_data, train_decision_stump, classify_decision_stump, 10, 5)
    result <- adaboost(tr_data, train_decision_stump,
                    classify_decision_stump, B, test_data)

    classifiers <- as.matrix(cbind(result$classifiers,result$alphas))

    test_y_prime <- aggregate_weak_classifiers(test_data$X, result$alphas, 
        result$classifiers, classify_decision_stump)
    test_error <- classifier_error(test_data$y, test_y_prime)

    errors <- list()
    errors$train <- result$training_errors
    errors$test <- result$test_errors
    errors$agg_test <- result$agg_test_errors
    errors$agg_train <- result$agg_training_errors

    return(list(classifiers=classifiers, errors=errors,
                final_error=test_error))
}

run_kfold_adaboost <- function(X, y, B, K) {
    # data <- cbind(data.frame(y),data.frame(X))
    idxs <- sample(1:nrow(X), nrow(X) * 0.8, replace=FALSE)
    test_data <- list(X=X[-idxs,], y=y[-idxs])
    train_data <- list(X=X[idxs,], y=y[idxs])
    tr_folds <- split_k_fold(train_data$X, train_data$y, K)

    errors <- list(test=matrix(nrow=0,ncol=B),train=matrix(nrow=0,ncol=B))

    classifiers <- list()
    best_class <- NA
    best_error <- 1

    pb <- create_progress_bar("text")
    pb$init(K)
    for(k in 1:K) {
        tr_fold <- list(X=subset(tr_folds$X, (tr_folds$folds != k)),
                        y=subset(tr_folds$y, (tr_folds$folds != k)))
        test_fold <- list(X=subset(tr_folds$X, (tr_folds$folds == k)),
                        y=subset(tr_folds$y, (tr_folds$folds == k)))

        result <- adaboost(tr_fold, train_decision_stump,
            classify_decision_stump, B, test_fold)

        classifiers <- c(classifiers,list(as.matrix(cbind(result$classifiers,result$alphas))))
        # max_idx <- order(result$agg_test_errors)[1]
        if(result$agg_test_errors[length(result$agg_test_errors)] < best_error) {
            best_class <- list(classifiers=result$classifiers,
                                alphas=result$alphas)
            best_error <- result$agg_test_errors
        }

        length(result$agg_test_errors) <- B
        length(result$agg_training_errors) <- B
        errors$test <- rbind(errors$test,result$agg_test_errors)
        errors$train <- rbind(errors$train,result$agg_training_errors)

        pb$step()
    }

    test_data$y_prime <- aggregate_weak_classifiers(test_data$X, best_class$alphas, 
                        best_class$classifiers, classify_decision_stump)
    test_data$error <- classifier_error(test_data$y, test_data$y_prime)
    return(list(errors=errors, classifiers=classifiers,
                best_class=best_class, final_error=test_data$error))
}

# Implementation of simple iterative adaboost algorithm
# X: training data (row is a data point)
# y: training classes
# train: training method for weak learner
# classify: classification method for weak learner
# B: number of weak learners to train
adaboost <- function(train_data, train, classify, B, test_data=list()) {
    X <- train_data$X
    y <- train_data$y
    n <- nrow(X)
    # initialize weights to even distribution
    w <- rep(1/n,n)

    # initialize returns
    classifiers <- matrix(nrow=0,ncol=3)
    training_errors <- vector()
    alphas <- vector()
    test_errors <- vector()
    agg_training_errors <- vector()
    agg_test_errors <- vector()

    pb <- create_progress_bar("text")
    pb$init(B)
    for(b in 1:B) {
        # train a weak learner on the weighted data
        classifier <- train(X, w, y)
        # classify data with weak learner
        y_prime <- classify(X, classifier)
        # compute the error of the weak learner
        error <- classifier_error(y, y_prime, w=w)

        # compute voting weights
        alpha <- log(1/error) - .1 * ifelse(b>1,alphas[b-1],1)
        # check for no improvement
        # if(alpha == 0) break

        # recompute weights
        for(i in 1:n) {
            if(y[i] != y_prime[i]) w[i] <- w[i] * exp(alpha)
        }

        # store data
        classifiers <- rbind(classifiers, classifier)
        training_errors <- c(training_errors,error)
        alphas <- c(alphas,alpha)

        # single classifier test error
        if(length(test_data) > 0){
            y_prime <- classify(test_data$X, classifier)
            test_errors <- c(test_errors,classifier_error(test_data$y, y_prime))
            
            # aggregated training and test error
            y_prime <- aggregate_weak_classifiers(X, alphas, classifiers, classify)
            agg_training_errors <- c(agg_training_errors, classifier_error(y, y_prime))
            y_prime <- aggregate_weak_classifiers(test_data$X, alphas, classifiers, classify)
            agg_test_errors <- c(agg_test_errors, classifier_error(test_data$y, y_prime))
        }
        pb$step()
    }
    cat('\n')

    # return weights and parameters of weak learners
    return(list(classifiers=classifiers,alphas=alphas,
            training_errors=training_errors,test_errors=test_errors,
            agg_training_errors=agg_training_errors,
            agg_test_errors=agg_test_errors))
}


# X: training data (rows are vectors of points)
# alphas: denotes the vector of voting weights
# classifiers: contains the parameters of all the weak learners
# return: classification labels for data
aggregate_weak_classifiers <- function(X, alphas, classifiers, classify) {
    classifiers <- as.matrix(classifiers)
    n <- nrow(X)
    B <- nrow(classifiers)
    y_prime <- vector(mode='numeric', length=n)
    y_votes <- matrix(nrow=B, ncol=n)

    for(b in 1:B) {
        y_votes[b,] <- classify(X,classifiers[b,])
    }
    for(i in 1:n) {
        weighted_class <- 0
        for(b in 1:B) {
            weighted_class <- weighted_class + alphas[b] * y_votes[b,i]
        }
        y_prime[i] <- sign(weighted_class)
    }
    return(y_prime)
}

