X <- as.matrix(read.table('uspsdata.txt'))
y <- scan('uspscl.txt')

length(X[,1])

# reorder matrix columns based on 2nd row
test[,order(test[2,])]