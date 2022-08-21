rm(list = ls())

library(zoo) 
library(TTR) 
library(xts) 
library(quantmod)

getSymbols("TTM",src="yahoo",from="2004-09-27",to=Sys.Date()) TTM<-get("TTM")	
#因為台股股號是數字，用數字繪圖會有問題 chartSeries(TTM["2004-09-27::2018-01-05)"],theme="white") chartSeries(TTM["2004-09-27::2018-01-05)"][,6],theme="white")

ma_short <- matrix(data=NA, ncol = 20,nrow=3345) 
ma_long <- matrix(data=NA, ncol = 20,nrow=3345)

temp0 <- 0
for (i in 1:20) { 
  for( j in 21:40){
    ma_short[,i]<-runMean(TTM[,6],n=i) 
    ma_long[,j-20]<-runMean(TTM[,6],n=j)
    position<-Lag(ifelse(ma_short[,i]>ma_long[,j-20], 1,0)) 
    temp <- ROC(Cl(TTM))*position
    temp <- na.omit(temp) 
    temp <- mean(temp) 
    if(temp>temp0){
      temp0 <- temp 
      result_i <- i 
      result_j <- j
    }
  }
}
temp0 
result_i 
result_j



ma_short<-runMean(TTM[,6],n=result_i) ma_long<-runMean(TTM[,6],n=result_j)

addTA(ma_short,on=1,col="blue") addTA(ma_long,on=1,col="red") addBBands()
