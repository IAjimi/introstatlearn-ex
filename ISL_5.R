####AN INTRODUCTION TO STATISTICAL LEARNING
####Chapter 5: Resampling Methods####
###Question 5. In Chapter 4, we used logistic regression to predict the probability of
#default using income and balance on the Default data set. We will
#now estimate the test error of this logistic regression model using the
#validation set approach. 
#(a) Fit a logistic regression model that uses income and balance to
#predict default.
model <- glm(default ~ income + balance, data = Default, family = binomial)
summary(model)

#(b) Using the validation set approach, estimate the test error of this
#model. In order to do this, you must perform the following steps:
#i. Split the sample set into a training set and a validation set.
shuffle <- sample(nrow(Default), nrow(Default))
train <- shuffle[1:(0.8*nrow(Default))]
test <- shuffle[(1+0.8*nrow(Default)):nrow(Default)]

#ii. Fit a multiple logistic regression model using only the training observations.
model <- glm(default ~ income + balance, data = Default[train, ], family = binomial)

#iii. Obtain a prediction of default status for each individual, using a threshold of 0.5
#NOTE: contrasts (Default$default) shows that Yes is coded as 1
model_preds <- predict(model, Default[test, ], type = "response") 
model_preds[model_preds > 0.5] <- "Yes" 

model_preds[model_preds < 0.5] <- "No"

#iv. Compute the validation set error
sum(model_preds == Default$default) / length(Default$default)

#(c) Repeat the process in (b) three times, using three different splits
#of the observations into a training set and a validation set. Comment
#on the results obtained.
set.seed(2)

error <- rep(0, 3)

for (i in 1:3) {
  shuffle <- sample(nrow(Default), nrow(Default))
  train <- shuffle[1:(0.8*nrow(Default))]
  test <- shuffle[(1+0.8*nrow(Default)):nrow(Default)]
  
  model <- glm(default ~ income + balance, data = Default[train, ], family = binomial)
  
  model_preds <- predict(model, Default[test, ], type = "response") 
  model_preds[model_preds > 0.5] <- "Yes" 
  model_preds[model_preds < 0.5] <- "No"
  
  error[i] <- sum(model_preds == Default$default) / length(Default$default)
}

error
#there is some variation in error ratesm though relatively small

#(d) Now consider a logistic regression model that predicts the probability
#of default using income, balance, and a dummy variable
#for student. Estimate the test error for this model using the validation
#set approach. Comment on whether or not including a
#dummy variable for student leads to a reduction in the test error
#rate.
set.seed(1)
shuffle <- sample(nrow(Default), nrow(Default))
train <- shuffle[1:(0.8*nrow(Default))]
test <- shuffle[(1+0.8*nrow(Default)):nrow(Default)]

model <- glm(default ~ income + balance + student, data = Default[train, ], family = binomial)

model_preds <- predict(model, Default[test, ], type = "response") 
model_preds[model_preds > 0.5] <- "Yes" 
model_preds[model_preds < 0.5] <- "No"

sum(model_preds == Default$default) / length(Default$default) #slightly worse

#6. We continue to consider the use of a logistic regression model on the
#Default data set. In particular, we will now compute estimates for
#the standard errors of the income and balance logistic regression coefficients
#using the bootstrap, and using the standard formula for computing the standard errors in the 
#glm() function. 
#(a) Using the summary() and glm() functions, determine the estimated
#standard errors for the coefficients associated with income and balance 
set.seed(1)
shuffle <- sample(nrow(Default), nrow(Default))
train <- shuffle[1:(0.8*nrow(Default))]
test <- shuffle[(1+0.8*nrow(Default)):nrow(Default)]

model <- glm(default ~ income + balance, data = Default[train, ], family = binomial)

summary(model)

#(b) Write a function, boot.fn(), that takes as input the Default data
#set as well as an index of the observations, and that outputs
#the coefficient estimates for income and balance in the multiple
#logistic regression model.
boot.fn <- function(data, index) {
  model2 <- glm(default ~ income + balance, data = data[index, ], family = binomial)
  model2$coefficients
}

boot.fn(Default, train)

#(c) Use the boot() function together with your boot.fn() function to
#estimate the standard errors of the logistic regression coefficients
#for income and balance.
boot(Default, boot.fn, R = 100)

#(d) Comment on the estimated standard errors obtained using the
#glm() function and using your bootstrap function.
#The standard errors are reasonably close.

#7. cv.glm() function can be used in order to compute the LOOCV test error estimate. 
#Alternatively, one could compute those quantities using just the glm() and
#predict.glm() functions, and a for loop. 
#(a) Fit a logistic regression model that predicts Direction using Lag1
#and Lag2.
model <- glm(Direction ~ Lag1 + Lag2, data = Weekly, family = binomial)
preds <- predict(model, Weekly)
preds[preds > 0.5] <- "Up"
preds[preds <= 0.5] <- "Down"

mean(preds != Weekly$Direction) #0.542 error rate

#(b) Fit a logistic regression model that predicts Direction using Lag1
#and Lag2 using all but the first observation.
model <- glm(Direction ~ Lag1 + Lag2, data = Weekly[-1, ], family = binomial)

#(c) Use the model from (b) to predict the direction of the first observation.
#You can do this by predicting that the first observation
#will go up if P(Direction="Up"|Lag1, Lag2) > 0.5. Was this observation
#correctly classified?
preds <- predict(model, Weekly[1, ])
preds[preds > 0.5] <- "Up"
preds[preds <= 0.5] <- "Down"

table(preds, Weekly$Direction[1])
#Yes it was

#(d) Write a for loop from i = 1 to i = n, where n is the number of
#observations in the data set, that performs each of the following
#steps:
#i. Fit a logistic regression model using all but the ith observation
#to predict Direction using Lag1 and Lag2.

#ii. Compute the posterior probability of the market moving up
#for the ith observation.
#iii. Use the posterior probability for the ith observation in order
#to predict whether or not the market moves up.
#iv. Determine whether or not an error was made in predicting
#the direction for the ith observation. If an error was made,
#then indicate this as a 1, and otherwise indicate it as a 0.
error <- rep(0, nrow(Weekly))

for (i in 1:nrow(Weekly)){
  model <- glm(Direction ~ Lag1 + Lag2, data = Weekly[-i, ], family = binomial)
  preds <- predict(model, Weekly[i, ])
  preds[preds > 0.5] <- "Up"
  preds[preds <= 0.5] <- "Down"
  
  error[i] <- c(0,0,1)[2 + (preds != Weekly$Direction[i])] #a bit weird but basically uses 
  #the fact that true = 1 and false = 0 to subsect the vector c(0, 0, 1) which assigns
  #to error 0/1 depending on value of boolean
}

#(e) Take the average of the n numbers obtained in (d)iv in order to
#obtain the LOOCV estimate for the test error. Comment on the
#results.
mean(error) #0.545 error rate
#pretty close to our initial results, tho they underestimated the error rate

#8. We will now perform cross-validation on a simulated data set.
#(a) Generate a simulated data set as follows:
set.seed(1)
y <- rnorm(100)
x <- rnorm(100)
y <- x-2*x^2+ rnorm(100)
#In this data set, what is n and what is p? Write out the model
#used to generate the data in equation form.
#n = 100, p = 100
#y = (x-2)*X^2 + e
#y = x^3 - 2x^2 + e
#(b) Create a scatterplot of X against Y . Comment on what you find.
plot(y,x)
#it's very obviously a non-linear relationship

#(c) Set a random seed, and then compute the LOOCV errors that
#result from fitting the following four models using least squares:
random_df <- data.frame(y, x)

#  i. Y = ??0 + ??1X + e
#  ii. Y = ??0 + ??1X + ??2X2 + e
#  iii. Y = ??0 + ??1X + ??2X2 + ??3X3 + e
#  iv. Y = ??0 + ??1X + ??2X2 + ??3X3 + ??4X4 + e

errors <- rep(0, 4)

for (i in 1:4){
  glm_fit <- glm(y ~ (poly(x, i)), data = random_df)
  cv_err <- cv.glm(random_df, glm_fit)
  errors[i] <- cv_err$delta[1] 
}
#uses automated glm LOOCV + loop to easily test all four models
#most dramatic change happens with ii, but best overall performance is iii
#not surprising bc we know the true model has X^3
