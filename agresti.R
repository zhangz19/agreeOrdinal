
rm(list=ls())
option <- as.character(1:5)
dat <- as.table(matrix(c( 22,  2,  2,  0,  0, 
                           5,  7, 14,  0,  0,
                           0,  2, 36,  0,  0,
                           0,  1, 14,  7,  0,
                           0,  0,  3,  0,  3), nrow  = 5, 
                 dimnames = list(classifier1 = option,
                 classifier2 = option)))
(dat <- t(dat))

r <- nrow(dat)
u <- c(1:r)

### ------------- model (2.4)
LambdaA <- LambdaB <- matrix(0, r^2, r)
mat <- matrix(0, r^2, 3)
Y <- numeric(r^2)
k <- 1
for(i in 1:r){
  for(j in 1:r){
     LambdaA[k,i] <- LambdaB[k,j] <- 1 
     mat[k,1] <- u[i]*u[j]; mat[k,2] <- 1*(i==j); mat[k,3] <- 1
     Y[k] <- dat[i,j]
     k <- k + 1
  }
}

X <- cbind(mat,LambdaA[,-r],LambdaB[,-r])

## put on the restrictions
l1 <- LambdaA[,-r]; l1[which(!rowSums(l1)),] <- -1
l2 <- LambdaB[,-r]; l2[which(!rowSums(l2)),] <- -1
X <- cbind(mat,l1,l2)
cbind(X,Y)
## look at the first two: beta & delta
summary(fit <- glm(Y~X-1, family = 'poisson'))
Yhat <- round(fitted(fit),3)
cbind(Yhat, Y)
# ---------------------------------------------------------------------------



### matrix in the way in the paper
X1 <- rbind(c(1,0,0,0, 8, 1), c(0,1,0,0, 6, 0), c(0,0,1,0, 4, 0), c(0,0,0,1, 2, 0),
            c(1,0,0,0, 4, 0), c(0,1,0,0, 3, 1), c(0,0,1,0, 2, 0), c(0,0,0,1, 1, 0),
            c(1,0,0,0, 0, 0), c(0,1,0,0, 0, 0), c(0,0,1,0, 0, 1), c(0,0,0,1, 0, 0),
            c(1,0,0,0,-4, 0), c(0,1,0,0,-3, 0), c(0,0,1,0,-2, 0), c(0,0,0,1,-1, 1),
            c(1,0,0,0,-8,-1), c(0,1,0,0,-6,-1), c(0,0,1,0,-4,-1), c(0,0,0,1,-2,-1)
)

COUNT <- t(matrix(
c( 22,  2,  2,  0,  0, 
   5,  7, 14,  0,  0,
   0,  2, 36,  0,  0,
   0,  1, 14,  7,  0,
   0,  0,  3,  0,  3), nrow  = 5))

A <- matrix(rep(1:5, 5), nrow  = 5)
B <- matrix(rep(1:5, each=5), nrow  = 5)

library(nnet)
(multinom(B ~ 1, weights=COUNT))




### test the Log-linear symmetry model 
#---------------------- MODEL 3: Log-linear Symmetry Model  
#-- see T.Meiser's paper "Loglinear Symmetry and Quasi-Symmetry Models
# for the Analysis of Change" for details.
#-- coding by Zhen "Thu Sep 22 14:53:27 2011"

Boys <- t(as.table(matrix(c(  
                        15, 18,  5, 
                        29, 54, 26, 
                        12, 48, 52
                 ), nrow = 3, dimnames = list(Mood1957 = 1:3,
                 Mood1955 = 1:3))))

Girls <- t(as.table(matrix(c(  
                        22, 21,  9, 
                        20, 56, 29, 
                         9, 44, 48
                 ), nrow = 3, dimnames = list(Mood1957 = 1:3,
                 Mood1955 = 1:3))))

logLinearSym <- function(dat, trend="none"){
r <- nrow(dat)
# construct the data and design matrix for glm
ind <- Y <- numeric(0); R <- (r+1)*r/2
mat <- cbind(1,diag(R)); mat <- mat[,-ncol(mat)]
for(i in 1:r){
  for(j in i:r){ Y <- c(Y, dat[i,j]); ind <- rbind(ind, c(i,j)) }
}
equalij <- which(ind[,1] == ind[,2])
mat[tmpind <- nrow(mat),equalij[-length(equalij)]+1] <- -1
for(i in 2:r){
  for(j in 1:(i-1)){ Y <- c(Y, dat[i,j]); mat <- rbind(mat, mat[which(ind[,1]==j & ind[,2]==i),]) }
}
if(trend == "downward") {
  mat <- cbind(mat,0); mat[(tmpind+1):nrow(mat),ncol(mat)] <- 1
}
if(trend == "upward") {
  indvec <- 1:tmpind
  mat <- cbind(mat,0); mat[indvec[-equalij],ncol(mat)] <- 1
}

tmp <- summary(fit <- glm(Y~mat-1, family = 'poisson'))
print(c(tmp$deviance, tmp$df.residual))
}

logLinearSym(Boys)
logLinearSym(Girls)
logLinearSym(Boys, trend='upward')
logLinearSym(Boys, trend='downward')

## test the Kappa coefficient in Agresti 2002  P432
option <- as.character(1:4)
dat <- as.table(matrix(c( 22,  2,  2,  0, 
                           5,  7, 14,  0,
                           0,  2, 36,  0,
                           0,  1, 17, 10), nrow  = 4, 
                 dimnames = list(classifier1 = option,
                 classifier2 = option)))
(dat <- t(dat))
source('utils.R')

library(coin)
library(e1071)
getTable(dat, model=0, option=option)


## test the results in P431 Agresti 2002
# data is in Table 10.5
option <- as.character(1:4)
dat <- as.table(matrix(c( 144,  2,  0,  0, 
                           5,  7, 14,  0,
                           0,  2, 36,  0,
                           0,  1, 17, 10), nrow  = 4, 
                 dimnames = list(classifier1 = option,
                 classifier2 = option)))
(dat <- t(dat))
source('utils.R')

library(coin)
library(e1071)
getTable(dat, model=0, option=option)

# NOT RUN
