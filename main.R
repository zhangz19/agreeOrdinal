## updated: "Tue Sep 20 21:46:43 2011"

rm(list=ls())

library(coin)

source('utils.R')

dat <- dataInput()

tab.names <-  names(sapply(dat, names)) 
out <- getTable(dat[[1]], model=c(1:5))
for(i in 2:length(tab.names)) 
  out <- rbind(out, getTable(dat[[i]], model=c(1:5)))

out <- as.data.frame(round(out,4))
rownames(out) <- tab.names

write.csv(out, file='out.csv',row.names=T)


# show example tables:
# good results: tn.C2.LPSAI with significant coefficients (high agreement)
index <- which(tab.names == 'tn.C2.LPSAI')
print(round(addmargins(dat[[index]])/sum(dat[[index]]),3))
getTable(dat[[index]], model=1)

# good results: tn.C2.LPSAI with nonsignificant coefficients (low agreement)
index <- which(tab.names == 'tp.C2.L2')
print(addmargins(dat[[index]]))
getTable(dat[[index]], model=1)

# not so good (low G square) with significant coefficients (high agreement)
index <- which(tab.names == 'sedi.C2.LPSAI')
print(addmargins(dat[[index]]))
getTable(dat[[index]], model=1)

# no model fit but large kappa
index <- which(tab.names == 'tp.L2.LPSAI')
print(addmargins(dat[[index]]))
getTable(dat[[index]], model=1)

# no model fit and small kappa
index <- which(tab.names == 'tp.LPSAI.LPUAI')
print(addmargins(dat[[index]]))
getTable(dat[[index]], model=1)

getTable(dat[[6]], model=1)
# NOT RUN
