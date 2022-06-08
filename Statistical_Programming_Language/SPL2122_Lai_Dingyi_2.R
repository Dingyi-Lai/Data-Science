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

### Exercise 2 -----------------------------------------------------------------
rm(list = ls())

getwd()
setwd("/Users/aubrey/Desktop/HU Berlin/SPL")
getwd()

# a) Read the data file
temp <- read.csv("highway.csv")
highway <- temp[,-1]

# b) Rename the variable adt to average daily traffic count in thousands.
colnames(highway)[3] <- c("average daily traffic count in thousands")
colnames(highway) # for check

# Analyse the highway dataset
attach(highway)

# c) What is the percentage of missing values in slim ?
sum(is.na(slim))/length(slim) # There is no missing values in slim
# So the answer is 0%

# d) Compute the mean and the interquartile range of the variable lane. 
# Create a table with absolute frequencies of the variable htype.
mean(lane) # mean of the variable lane
IQR(lane) # IQR of the variable lane
table(htype)

# e) What is the percentage of observations in slim which are greater than 70? 
# What is the percentage of observations in htype which have the values MA, PA or FAI?
length(slim[slim>70])/length(slim) # There is no observation in slim
# that is greater than 70, so its percentage is 0

round(length(htype[!(htype %in% c('MA', 'PA','FAI'))])/length(htype),4)
# The percentage of observations in htype which have the values MA, PA or FAI is around 5.13%

# f) Compute the Bravais-Pearson correlation between the variables itg and shld.
cor(itg,shld) # The default method is Bravais-Pearson correlation, and it's around 0.375

# g) Create a new categorical variable slim_cat from slim with 1 if the observed value is 
# smaller equal 50, 2 if the observed value is in (50; 65], and 3 if the observed value is
# larger than 65. Comment about the cut values; are they choosen sensibly?
# Create a contingency table from slim_cat and htype with the relative frequencies.
detach(highway)
temp <- rep(0, dim(highway)[1])
temp[highway$slim <= 50] <- 1
temp[(highway$slim > 50) & (highway$slim <= 65)] <- 2
temp[highway$slim > 65] <- 3
highway$slim_cat = temp

attach(highway)
table(slim_cat) 
# The cut value is not chosen sensibly. Because the frequency table shows that only 1 observation
# belongs to the third group (larger than 65)

# For the contingency table
prop.table(table(slim_cat, htype))

# h) Compute the 3. quartile and the interquartile range of the variable rate for each 
# subgroup defined by htype.
myFun <- function(x) {
  c(thirdQ = quantile(x, 3/4), IQR = IQR(x))
}

tapply(rate, htype, myFun)

# i) Run a statistical test on the percentage of FAI in htype. The null hypothesis should
# be H0 : pi >= 0.13. Verbalize the null hypothesis. What does the p-value tell you?

# The binomial test is used for a comparison between an observed percentage to a theoretical
# percentage: the binomial test
nFAI <- sum(htype == 'FAI')
n <- length(htype)
binom.test(nFAI,n,p = 0.13, alternative = 'less')
# The null hypothesis means the percentage of FAI in htype is greater equal to 0.13
# The p-value is 0.6031>0.05, which means that there is no sufficient evidence to reject the
# null hypothesis that this percentage is greater equal  to 0.13

# j) Run a simple linear regression with trks (dependent variable) and len (independent
# variable). How good is your regression? What does the regression slope tell you? 
# Do you think that this regression makes sense?
summary(lm(trks~len))
# Although the p-value in t-test about intercept and len and in F-test about the whole regression
# shows that they are all statistically significant, but the goodness of fit, indicated by 
# Multiple R-squared and Adjusted R-squared, looks not so satisfactory. According to Multiple R-squared,
# only 24.6% of trks can be interpreted by len.
# The slope tells me that 1 unit increase in length of the Highway1 segment in miles will result in 
# 0.15345 unit increase in truck volume as a percent of the total volume

# It make sense. Although the truck volume will not only be affected by the length of
# the Highway 1 segment, it could be the case that the longer one Highway, the more likely that it
# is used for long-haul transportation, where trucks with large volume more likely to be used