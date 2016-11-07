#---------------------- input data

dataInput <- function(option = c("high", "medium", "low")){

# Sediment_Table
tab.sedi.C2.L2 <- t(as.table(matrix(c(  
                        0,  2,  2, 
                        0,  8,  6, 
                        5, 27, 204
                 ), nrow = 3, dimnames = list(L2 = option,
                 C2 = option))))

tab.sedi.C2.LPSAI <- t(as.table(matrix(c(  
                        2,  0,  2, 
                        4,  8,  2, 
                        2, 11, 223
                 ), nrow = 3, dimnames = list(LPSAI = option,
                 C2 = option))))

tab.sedi.C2.LPUAI <- t(as.table(matrix(c(  
                        2,  1,  1, 
                        1, 11,  2, 
                       13, 21, 202
                 ), nrow = 3, dimnames = list(LPUAI = option,
                 C2 = option))))

tab.sedi.L2.LPSAI <- t(as.table(matrix(c(  
                        0,  0,  5, 
                        7, 10, 20, 
                        1,  9, 202
                 ), nrow = 3, dimnames = list(LPSAI = option,
                 L2 = option))))

tab.sedi.L2.LPUAI <- t(as.table(matrix(c(  
                        0,  0,  5, 
                       10, 12, 15, 
                        6, 21, 185
                 ), nrow = 3, dimnames = list(LPUAI = option,
                 L2 = option))))

tab.sedi.LPSAI.LPUAI <- t(as.table(matrix(c(  
                        5,  3,  0, 
                        5, 14,  0, 
                        6, 16, 205
                 ), nrow = 3, dimnames = list(LPUAI = option,
                 LPSAI= option))))





# TN_Table
tab.tn.C2.L2 <- t(as.table(matrix(c(  
                        0,  2, 31, 
                        4, 10, 50, 
                        1,  0, 156
                 ), nrow = 3, dimnames = list(L2 = option,
                 C2 = option))))

tab.tn.C2.LPSAI <- t(as.table(matrix(c(  
                        7, 10, 16, 
                        1, 15, 48, 
                        0,  5, 152
                 ), nrow = 3, dimnames = list(LPSAI = option,
                 C2 = option))))

tab.tn.C2.LPUAI <- t(as.table(matrix(c(  
                        2, 24,  7, 
                        6, 17, 41, 
                        2,  4, 151
                 ), nrow = 3, dimnames = list(LPUAI = option,
                 C2 = option))))

tab.tn.L2.LPSAI <- t(as.table(matrix(c(  
                        0,  0,  5, 
                        1,  0, 11, 
                        7, 30, 200
                 ), nrow = 3, dimnames = list(LPSAI = option,
                 L2 = option))))

tab.tn.L2.LPUAI <- t(as.table(matrix(c(  
                        0,  0,  5, 
                        1,  2,  9, 
                        9, 43, 185
                 ), nrow = 3, dimnames = list(LPUAI = option,
                 L2 = option))))

tab.tn.LPSAI.LPUAI <- t(as.table(matrix(c(  
                        3,  5,  0, 
                        3, 20,  7, 
                        4, 20, 192
                 ), nrow = 3, dimnames = list(LPUAI = option,
                 LPSAI= option))))



# TP_Table
tab.tp.C2.L2 <- t(as.table(matrix(c(  
                        0,  4, 21, 
                        0,  0, 14, 
                        6, 36, 173
                 ), nrow = 3, dimnames = list(L2 = option,
                 C2 = option))))

tab.tp.C2.LPSAI <- t(as.table(matrix(c(  
                        0,  0, 25, 
                        0,  0, 14, 
                        9, 20, 186
                 ), nrow = 3, dimnames = list(LPSAI = option,
                 C2 = option))))

tab.tp.C2.LPUAI <- t(as.table(matrix(c(  
                        1, 20,  4, 
                        1,  3, 10, 
                        7, 17, 191
                 ), nrow = 3, dimnames = list(LPUAI = option,
                 C2 = option))))

tab.tp.L2.LPSAI <- t(as.table(matrix(c(  
                        6,  0,  0, 
                        3, 20, 17, 
                        0,  0, 208
                 ), nrow = 3, dimnames = list(LPSAI = option,
                 L2 = option))))

tab.tp.L2.LPUAI <- t(as.table(matrix(c(  
                        0,  0,  6, 
                        0,  2, 38, 
                        9, 38, 161
                 ), nrow = 3, dimnames = list(LPUAI = option,
                 L2 = option))))

tab.tp.LPSAI.LPUAI <- t(as.table(matrix(c(  
                        0,  0,  9, 
                        0,  0, 20, 
                        9, 40, 176
                 ), nrow = 3, dimnames = list(LPUAI = option,
                 v = option))))

dat <- list(
  'sedi.C2.L2' = tab.sedi.C2.L2, 
  'sedi.C2.LPSAI' = tab.sedi.C2.LPSAI, 
  'sedi.C2.LPUAI' = tab.sedi.C2.LPUAI, 
  'sedi.L2.LPSAI' = tab.sedi.L2.LPSAI, 
  'sedi.L2.LPUAI' = tab.sedi.L2.LPUAI, 
  'sedi.LPSAI.LPUAI' = tab.sedi.LPSAI.LPUAI,
  'tn.C2.L2' = tab.tn.C2.L2, 
  'tn.C2.LPSAI' = tab.tn.C2.LPSAI, 
  'tn.C2.LPUAI' = tab.tn.C2.LPUAI, 
  'tn.L2.LPSAI' = tab.tn.L2.LPSAI, 
  'tn.L2.LPUAI' = tab.tn.L2.LPUAI, 
  'tn.LPSAI.LPUAI' = tab.tn.LPSAI.LPUAI,
  'tp.C2.L2' = tab.tp.C2.L2, 
  'tp.C2.LPSAI' = tab.tp.C2.LPSAI, 
  'tp.C2.LPUAI' = tab.tp.C2.LPUAI, 
  'tp.L2.LPSAI' = tab.tp.L2.LPSAI, 
  'tp.L2.LPUAI' = tab.tp.L2.LPUAI, 
  'tp.LPSAI.LPUAI' = tab.tp.LPSAI.LPUAI
)

  return(dat)
}


# Agresti 2002, P435: Kappa coefficients and Weighted Kappa
# "Mon Oct 03 02:14:44 2011" by Zhen@CSTAT
# kappa is for nomial while weighted kappa for ordinal
# weights = 2: interpretable as intra-class correlation coefficient
# weights = 1: directly related to Cicchetti's C statistics (Cicchetti 1972)
myKappa <- function(dat, weights = 2){
  n <- sum(dat); Pd <- diag(dat)/n; Pc <- colSums(dat)/n; Pr <- rowSums(dat)/n
  P0 <- sum(Pd); Pe <- sum(Pc * Pr)
  kappa <- (P0 - Pe) / (1 - Pe) 
  tsum <- 0
  for(a in 1:(r <- nrow(dat))){
     for(b in 1:r){ tsum <- tsum + dat[a,b]/n*(Pr[b] + Pc[a])^2 }
  }
  kappa.var <- 1/n*(P0*(1-P0)/(1-Pe)^2 + 2*(1-P0)*(2*P0*Pe - sum(Pd*(Pc + Pr)))/(1-Pe)^3 +
                     (1-P0)^2*(tsum - 4*Pe^2)/(1-Pe)^4)
  kappa.std <- sqrt(kappa.var)

  weighted.agree <- s1 <- 0; Wmat <- matrix(0,r,r)
  for(a in 1:r){
    for(b in 1:r){ 
      if(weights == 1) w <- 1 - abs(a-b)/(r-1)
      if(weights == 2) w <- 1 - (a-b)^2/(r-1)^2 # squared weight suggested by Fleiss, Cohen (1973)
      weighted.agree <- weighted.agree + w*dat[a,b]/n
      s1 <- s1 + w*Pc[a]*Pr[b] 
      Wmat[a,b] <- w
    }
  }
  weighted.kappa <- (weighted.agree - s1) / (1 - s1) # Cohen (1968) Weighted kappa
  
  # now calculate the large sample standard error of weighted kappa
  # Fleiss, Cohen & Everitt, 1969 
  Widot <-  as.vector(Wmat %*% Pc); Wdotj <- as.vector(Wmat %*% Pr)
  Widotmat <- matrix(rep(Widot, r), ncol=r)
  Wdotjmat <- matrix(rep(Wdotj, r), ncol=r, byrow=T)
  A <- (1+s1) - sum(dat/n * Wmat * (Widotmat + Wdotjmat))
  B <- (1+s1)^2 - sum(dat/n * (Widotmat + Wdotjmat)^2)
  C <- 1 - sum(dat/n * Wmat^2)
  wkappa.var <- 1/(n*(1-s1)^2)*(2*A*(1-weighted.kappa) - B*(1-weighted.kappa)^2 - C)
  wkappa.std <- sqrt(wkappa.var)
 
  out <- as.data.frame(cbind(P0, kappa, kappa.std, kappa-kappa.std*qnorm(.975), kappa+kappa.std*qnorm(.975), 
                             weighted.agree, weighted.kappa, weighted.kappa-wkappa.std*qnorm(.975), weighted.kappa+wkappa.std*qnorm(.975) )); row.names(out) <- ''
  names(out) <- c('agreement','kappa','stdev','lower','upper','weighted.agreement', 'weighted.kappa','w.lower','w.upper')
  return(out)
}

getTable <- function(dat, model=c(1:5), u=NULL, option=c("high", "medium", "low")){

# dat is any square contigency table. 
if(nrow(dat)!=length(option)) option <- 1:nrow(dat)

#---------------------- TEST & Summary Statistics

#---------------------- Stuart-Maxwell test with dat treated as ordered
# Asymptotic Marginal-Homogeneity Test for Ordered Data
# The null hypothesis of independence of row and column totals is
# tested. Stuart's W_0 statistic(Stuart, 1955, Agresti, 2002, page 422, 
# also known as Stuart-Maxwell test) is computed. the Stuart-Maxwell test 
#(Stuart, 1955; Maxwell, 1970; Everitt, 1977) tests marginal homogeneity for 
# all categories simultaneously. 

mh.pvalue <- pvalue(mh_test(dat, scores = list(response = 1:length(option))))


#---------------------- McNemar's Chi-squared test
# Performs McNemar's chi-squared test for symmetry of rows and
# columns in a two-dimensional contingency table.
# The null is that the probabilities of being classified into cells
# '[i,j]' and '[j,i]' are the same.

# mcnemar.test(dat)


#---------------------- Kappa coefficients of agreement 
# see "?classAgreement" for details after library(e1071)
# kappa.coef <- classAgreement(dat)$kappa

# use the function myKappa
out <- cbind(myKappa(dat), mh.pvalue)


#----------------------  Modeling part
r <- nrow(dat)
if(is.null(u)) u <- c(1:r)  # uniform association

LambdaA <- LambdaB <- matrix(0, r^2, r)
mat <- matrix(0, r^2, 3) # beta, delta, intercept
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
# print(cbind(X,Y))

## put on the restrictions

if(1 %in% model){
#---------------------- MODEL 1: Agreement plus linear-by-linear association (A.Agresti, 1988)
#-- see A.Agresti's paper "A model for agreement between ratings on an 
#--   ordinal scale" for details.
#-- coding by Zhen "Wed Sep 07 03:23:17 2011"

## look at the first two: beta & delta
tmp <- summary(fit <- glm(Y~X-1, family = 'poisson'))
Yhat <- round(fitted(fit),3)
cat('observed:\n'); print(vec2mat(Y,r,option))
cat('fitted:\n'); print(vec2mat(Yhat,r,option))
fit.coef <- as.data.frame(t(as.vector(t(tmp$coefficients[1:2, c(1,4)]))))
names(fit.coef) <- c('beta','beta.pvalue', 'delta', 'delta.pvalue')
# for uniform association: interested tau: exp(beta + 2*delta)
indisting.para <- as.numeric(exp(fit.coef[1]*(u[2]-u[1])^2 + 2*fit.coef[3]))
G2 <- tmp$deviance
resid.df <- tmp$df.residual
AICs <- tmp$aic
Pvalue <- 1 - pchisq(G2, resid.df)

tmpout <- cbind(as.data.frame(fit.coef), indisting.para, G2, resid.df, AICs, Pvalue )
names(tmpout) <- paste('M1', names(tmpout), sep='.')
out <- cbind(out, tmpout)
}


if(2 %in% model){
#---------------------- MODEL 2: Quasi-independence Model(P431 A.Agresti 2002)
# Agreement association only
#-- coding by Zhen "Wed Sep 21 01:08:18 2011"
## look at the first two: beta & delta
tmp <- summary(fit <- glm(Y~X[,-1]-1, family = 'poisson'))
fit.coef <- as.data.frame(t(as.vector(t(tmp$coefficients[1, c(1,4)]))))
names(fit.coef) <- c('delta', 'delta.pvalue')
# for uniform association: interested tau: exp(2*delta)
indisting.para <- as.numeric(exp(2*fit.coef[1]))
G2 <- tmp$deviance
resid.df <- tmp$df.residual
AICs <- tmp$aic
Pvalue <- 1 - pchisq(G2, resid.df)

tmpout <- cbind(as.data.frame(fit.coef), indisting.para, G2, resid.df, AICs, Pvalue)
names(tmpout) <- paste('M2', names(tmpout), sep='.')
out <- cbind(out, tmpout)
}


if(3 %in% model){
#---------------------- MODEL 3: Agreement plus linear-by-linear association (A.Agresti, 1988)
#-- Comparing to MODEL 1, delta(i,j) = delta_i I(i==j) stead of single parameter delta 
#-- coding by Zhen "Wed Sep 07 03:23:17 2011"

mat2 <- matrix(0,nrow(mat),r); ind2 <- which(mat[,2]==1)
for(i in 1:r) mat2[ind2[i],i] <- 1
mat2 <- cbind(mat[,1], mat2, mat[,3])
X3 <- cbind(mat2,LambdaA[,-r],LambdaB[,-r])  #intercept, beta, detal1, ..., deltar

## look at the first two: beta & delta
tmp <- summary(fit <- glm(Y~X3-1, family = 'poisson'))
cbind(round(fitted(fit),3),Y)
fit.coef <- as.data.frame(t(as.vector(t(tmp$coefficients[1:(r+1), c(1,4)]))))
nam <- numeric(); for(i in 1:r) nam <- c(nam, paste(c('delta','delta.pvalue'),i,sep=''))
names(fit.coef) <- c('beta','beta.pvalue', nam)
G2 <- tmp$deviance
resid.df <- tmp$df.residual
AICs <- tmp$aic
Pvalue <- 1 - pchisq(G2, resid.df)

tmpout <- cbind(as.data.frame(fit.coef), G2, resid.df, AICs, Pvalue)
names(tmpout) <- paste('M3', names(tmpout), sep='.')
out <- cbind(out, tmpout)
}



if(4 %in% model | 5 %in% model){
#---------------------- MODEL 4: Log-linear Symmetry Model and with trend  
#-- see T.Meiser's paper "Loglinear Symmetry and Quasi-Symmetry Models
# for the Analysis of Change" for details.
#-- coding by Zhen "Thu Sep 22 14:53:27 2011"

# construct the data and design matrix for glm
ind <- Y <- numeric(0); R <- (r+1)*r/2
mat <- cbind(1,diag(R)); mat <- mat[,-ncol(mat)]
for(i in 1:r){
  for(j in i:r){ Y <- c(Y, dat[i,j]); ind <- rbind(ind, c(i,j)) }
}
equalij <- which(ind[,1] == ind[,2])
mat[tmpind <- nrow(mat),equalij[-length(equalij)]+1] <- -1
for(i in 2:r){
  for(j in 1:(i-1)){
    Y <- c(Y, dat[i,j]); mat <- rbind(mat, mat[which(ind[,1]==j & ind[,2]==i),])
  }
}

if(4 %in% model){ #Log-linear Symmetry Model
tmp <- summary(fit <- glm(Y~mat-1, family = 'poisson'))
G2 <- tmp$deviance
resid.df <- tmp$df.residual
AICs <- tmp$aic
Pvalue <- 1 - pchisq(G2, resid.df)

tmpout <- as.data.frame(cbind(G2, resid.df, AICs, Pvalue))
names(tmpout) <- paste('M4', names(tmpout), sep='.')
out <- cbind(out, tmpout)
}

if(5 %in% model){ #Log-linear Symmetry Model with trend 
mat <- cbind(mat,0); mat[(tmpind+1):nrow(mat),ncol(mat)] <- 1 #downward trend
tmp <- summary(fit <- glm(Y~mat-1, family = 'poisson'))
G2 <- tmp$deviance
resid.df <- tmp$df.residual
AICs <- tmp$aic
Pvalue <- 1 - pchisq(G2, resid.df)

tmpout <- as.data.frame(cbind(G2, resid.df, AICs, Pvalue))
names(tmpout) <- paste('M5', names(tmpout), sep='.')
out <- cbind(out, tmpout)
}

}
return(out)
}

# utils functions

# convert vector to matrix(table)
vec2mat <- function(vec, r, option){
  mat <- as.data.frame(matrix(vec,ncol=r, byrow=T))
  names(mat) <- option
  row.names(mat) <- option
  return(mat)
}
## NOT RUN

