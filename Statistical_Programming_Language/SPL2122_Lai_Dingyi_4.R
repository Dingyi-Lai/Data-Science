################################################################################ 
################## Statistical Programming Languages 2021/22 ###################
##################               Take-home Exam              ###################
##################                                           ###################
##################   	  Sigbert Klinke, Eva-Maria Maier,     ###################
##################       		  Alexander Volkmann             ###################
################################################################################

#-------------------------------------------------------------------------------
# Surname: Lai
# Name: Dingyi
# Student ID (Matrikelnummer): 615865
#-------------------------------------------------------------------------------

### Exercise 4 -----------------------------------------------------------------
rm(list = ls())

getwd()
setwd("/Users/aubrey/Desktop/HU Berlin/SPL")
getwd()

# a) Read the data file election08.csv into R.
temp <- read.csv("election08.csv")
election08 <- temp[,-1]

attach(election08)
# b) Construct a graphic similar to the following graphic below, which includes everything inside the
# thick black frame (except the frame itself). Use for the scatterplot for the x-axis the variable BA
# and the y-axis Income. As categorical variable use the variable ObamaWin.
pdf(file = "SPL2122_Lai_Dingyi_4.pdf", width = 8, height = 8) # Save pdf
layout(matrix(c(1,2,3,3), ncol=2, byrow=TRUE), heights=c(7, 1))

par(mai=rep(0.7, 4),mar = c(6,4,3,1) + 0.1)
col.vec <- c(2,4)
pch.vec <- c(2,4)
plot(BA, Income, col = col.vec, pch = pch.vec,
     xlab = "Percentage of adults with at least a college education",
     ylab = "Per capita income in the state as of 2007 (in dollars)", # axes labels
     cex.lab = 1.1, # The magnification of label
     cex.axis = 0.8, # The magnification of axis
     las = 1) # size of labels
df <- data.frame(Income, BA, ObamaWin)
boxplot(Income~ObamaWin, data = df,col = col.vec,
        xlab = "Obama (Democrat) or McCain (Republican) wins", 
        ylab = "Per capita income in the state as of 2007 (in dollars)", # axes labels
        cex.lab = 1.1, # The magnification of label
        xaxt='n', # remove x-axis
        cex.axis = 0.8, # The magnification of axis
        las = 1) # direction of label

par(mai=c(0,0,0,0))
plot.new()
legend(x = "center",inset = 0,
       legend = c("McCain", "Obama"), # 1=Obama, 0=McCain
       col=col.vec, cex=1, horiz = TRUE, pch = pch.vec,
       text.col =col.vec)

# Reference: https://stackoverflow.com/questions/12402319/centring-legend-below-two-plots-in-rlty=3
dev.off()
detach(election08)
