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

### Exercise 3 -----------------------------------------------------------------
rm(list = ls())

getwd()
setwd("/Users/aubrey/Desktop/HU Berlin/SPL")
getwd()

# a) Read the data file grain.csv into R.
temp <- read.csv("grain.csv")
grain <- temp[,-1]

# b) Convert the variable time to a posixct or Date variable.
# "POSIXct is used if handling of time zones and daylight saving is important"
grain$time <- as.POSIXct(grain$time)
str(grain)

attach(grain)
# c) Compute a exponential trend for Yield.of.grain on time. How good is your trend?
# Do you think that this trend estimation makes sense?
yog <- chartr(old = ",", new = ".", yield.of.grain) # Substitute '.' for ',' in a string
df <- data.frame(t = min(format(time, format="%Y")):max(format(time, format="%Y")), yog = as.numeric(yog))

et <- lm(log(yog)~t, data=df)
summary(et)

# Although the p-value in t-test about intercept and time and in F-test about the whole regression
# shows that they are all statistically significant, but the goodness of fit, indicated by 
# Multiple R-squared and Adjusted R-squared, looks not so satisfactory. According to Multiple R-squared,
# only 24.01% of Yield.of.grain can be interpreted by time.
# The slope tells me that 1 year going by results in -0.009307 unit decrease in Yield.of.grain,
# which makes sense potentially due to over-exploitation or soil erosion.

# d) Plot the time series and the trend from exercise c). Make sure that titles, axes, labels and
# so on have useful names and are clearly readable.
plot(df,
     xlab = "Time", ylab = "Yield.of.grain", # axes labels
     cex.lab = 1.1, # size of labels
     main = "The Exponential Trend of Yield.of.grain",
     col = "lightblue")
lines(df$t, exp(fitted(et)), col="blue")

detach(grain)
