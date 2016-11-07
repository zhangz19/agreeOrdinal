# IRLS algorithm for poisson family
myIRLSpois <- function(X,Y){
  mu <- Y; eta <- log(mu + 1e-16)
  dev_odd <- 0; dev_new <- 1; dev_dif <- 1
  tol <- 1e-5; iteration <- 0
  while(abs(dev_dif) > tol){
     W <- diag(as.vector(exp(eta)^2/mu))
     Z <- (Y - mu)/mu + eta
     beta <- chol2inv(chol(t(X)%*%W%*%X)) %*% t(X)%*%W%*%Z
     eta <- X%*%beta; mu <- exp(eta)
     dev_old <- dev_new
     dev_new <- 2*sum(Y*(log(Y+1e-16) - log(mu+1e-16)) - Y + mu)
     dev_dif <- dev_new - dev_old
     iteration <- iteration + 1
     print(dev_dif)
  }
  se <- matrix(sqrt(diag(chol2inv(chol(t(X)%*%W%*%X)))))
  mat <- as.data.frame(cbind(beta, se, beta/se))
  print(mat)
}
