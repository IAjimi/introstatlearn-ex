####Chapter 6####
#Question 8####
#In this exercise, we will generate simulated data, and will then use this data to perform 
#best subset selection.
library(ISLR)
library(leaps)

#(a) Use the rnorm() function to generate a predictor X of length n = 100, 
#as well as a noise vector  of length n = 100.
set.seed(1)
x <- rnorm(100)
e <- rnorm(100)

#(b) Generate a response vector Y of length n = 100 according to the model
#Y = ??0 + ??1X + ??2X^2 + ??3X^3 + e, where ??0, ??1, ??2, and ??3 are constants of your choice.
y <- 1 +2*x + 3*x^2 + 4*x^3 + e

#(c) Use the regsubsets() function to perform best subset selection in order to choose the 
#best model containing the predictors X, X^2,...,X^10. 
#What is the best model obtained according to Cp, BIC, and adjusted R2? 
#Show some plots to provide evidence and report the coefficients of the best model obtained.
sim_df <- data.frame(y, x)

reg <- regsubsets(y ~ poly(x, 10, raw = TRUE), sim_df, nvmax = 10) #important to use raw = TRUE
reg_sum <- summary(reg) 

which.max(reg_sum$adjr2) #biggest adj rsq -> 4
which.min(reg_sum$rss) #lowest rss -> 10
which.min(reg_sum$cp) #lowest cp -> 4
which.min(reg_sum$bic) #lowest bic -> 3 most accurate criterion

par(mfrow=c(2,2))
plot(reg_sum$rss ,xlab="Number of Variables ",ylab="RSS", type="l")
points(which.min(reg_sum$rss), reg_sum$rss[which.min(reg_sum$rss)], col="red",cex=2,pch =20)
plot(reg_sum$adjr2 ,xlab="Number of Variables ", ylab="Adjusted RSq",type="l")
points(which.max(reg_sum$adjr2), reg_sum$adjr2[which.max(reg_sum$adjr2)], col="red",cex=2,pch =20)
plot(reg_sum$cp ,xlab="Number of Variables ", ylab="Cp",type="l")
points(which.min(reg_sum$cp), reg_sum$cp[which.min(reg_sum$cp)], col="red",cex=2,pch =20)
plot(reg_sum$bic ,xlab="Number of Variables ", ylab="Bayesian Info. Criterion",type="l")
points(which.min(reg_sum$bic), reg_sum$bic[which.min(reg_sum$bic)], col="red",cex=2,pch =20)
####apologies for the ugly code#####

#either 4 or 3 seem like the best models -- pick 3 bc leaner model

coef(reg,id=3) 
coef(reg,id=4) 

#the variables kept are the 1st, 2nd, 3rd polynomials of x for 3 var
#model with 4 var includes the 5th polynomial of x, which is a bit weird and objectively wrong
#the coefficients for the model with 3 variables are very close to the true coefficients
#those of the 4var model are a bit off

#(d) Repeat (c), using forward and also using backwards stepwise selection. How does your answer 
#compare to the results in (c)?
reg_fwd <- regsubsets(y ~ poly(x, 10, raw = TRUE), sim_df, nvmax = 10, method = "forward") 
reg_fwd_sum <- summary(reg_fwd) 

which.max(reg_fwd_sum$adjr2) #biggest adj rsq -> 4
which.min(reg_fwd_sum$rss) #lowest rss -> 10
which.min(reg_fwd_sum$cp) #lowest cp -> 4
which.min(reg_fwd_sum$bic) #lowest bic -> 3 

reg_bck <- regsubsets(y ~ poly(x, 10, raw = TRUE), sim_df, nvmax = 10, method = "backward") 
reg_bck_sum <- summary(reg_bck) 

which.max(reg_bck_sum$adjr2) #biggest adj rsq -> 4
which.min(reg_bck_sum$rss) #lowest rss -> 10
which.min(reg_bck_sum$cp) #lowest cp -> 4
which.min(reg_bck_sum$bic) #lowest bic -> 3

coef(reg_fwd,id=3) 
coef(reg_bck,id=3)
#we get the same results as before with both forward and backward selection, same variables,
#same coefficients

#(e) Now fit a lasso model to the simulated data, again using X, X2, ...,X10 as predictors. 
#Use cross-validation to select the optimal value of ??. Create plots of the cross-validation 
#error as a function of ??. Report the resulting coefficient estimates, and discuss the results obtained.
library(glmnet)
x_iv <- model.matrix(y ~ poly(x, 10, raw = TRUE), sim_df)[ ,-1] #create x predictor matrix
cv.out <- cv.glmnet(x_iv, y_dv, alpha=1) 
plot(cv.out)
cv.out$lambda.min #the optimal lambda is of 0.0766

best_model <- glmnet(x_iv, y, alpha = 1)
lasso.pred <- predict(best_model, s= cv.out$lambda.min, type= "coefficients")
#the model picks the 1st, 2nd, 3rd, 4th, 5th, and 7th polynomials of x
#however, the coefficients of the 4th, 5th and 7th are relatively small
#the coeff of the intercept, 1st, 2nd and 3rd poly are close to the real values

#(f) Now generate a response vector Y according to the model Y = ??0 + ??7X7 + e,
#and perform best subset selection and the lasso. Discuss the results obtained.
y <- 1 + 2*x^7 + e
sim_df <- data.frame(y, x)

##using best subset
reg <- regsubsets(y ~ poly(x, 10, raw = TRUE), sim_df, nvmax = 10) 
reg_sum <- summary(reg_bck) 

which.max(reg_sum$adjr2) #biggest adj rsq -> 4
which.min(reg_sum$rss) #lowest rss -> 10
which.min(reg_sum$cp) #lowest cp -> 2 
which.min(reg_sum$bic) #lowest bic -> 1
#the results are ambiguous, and kind of all over the place

coef(reg_fwd,id=1) 
coef(reg_bck,id=2)
#for some reason the model with one variable does *not* include x^7
#the second model (2 var) is much better -- most accurate criterion here was cp

##with lasso
x_iv <- model.matrix(y ~ poly(x, 10, raw = TRUE), sim_df)[ ,-1] #create x predictor matrix
cv.out <- cv.glmnet(x_iv, y, alpha=1) 
plot(cv.out)
cv.out$lambda.min #the optimal lambda is of 3.88

best_model <- glmnet(x_iv, y, alpha = 1)
predict(best_model, s= cv.out$lambda.min, type= "coefficients")

#not only does the lasso model include the only variable of interest, but the coefficients are
#very close to our true values


###Question 9 ####
###In this exercise, we will predict the number of applications received using the other 
###variables in the College data set.
#Loading Libraries
library(boot)
library(glmnet)
library(glmnet)
library(pls)

#(a) Split the data set into a training set and a test set.
random <- sample(c(1:nrow(College)), nrow(College))
train <- random[1:(nrow(College)*(6/7))]
test <- random[(nrow(College)*(6/7)):nrow(College)]

#(b) Fit a linear model using least squares on the training set, and report the test error 
#obtained.
reg <- glm(Apps ~., data=College[train, ])
mean((predict(reg, College[test, ]) - College$Apps[test])^2)

coef(reg) #coefficients of model

#(c) Fit a ridge regression model on the training set, with ?? chosen by cross-validation. 
#Report the test error obtained.
x <- model.matrix(Apps ~., College)[ ,-1]
y <- College$Apps

set.seed(1) #set seed for reproduc.
reg_ridge <- cv.glmnet(x[train, ], y[train ], alpha=0) #performs ridge reg w 10fold cv
plot(reg_ridge)
bestlam <- reg_ridge$lambda.min #finds lambda w lowest train error, here 409 approx
ridge_pred <- predict(reg_ridge, s= bestlam, newx=x[test, ])
mean((ridge_pred - y[test])^2)
#the error of the ridge model is slightly higher

predict(reg_ridge, s= bestlam, type = "coefficients") #coefficients of best model


#(d) Fit a lasso model on the training set, with ?? chosen by crossvalidation.
#Report the test error obtained, along with the number of non-zero coefficient estimates.
set.seed(1)
reg_lass <- cv.glmnet(x[train, ], y[train], alpha = 1 ) #alpha = 1 -> lasso instead of ridge
plot(reg_lass)

lass_pred <- predict(reg_lass, s = reg_lass$lambda.min, newx = x[test, ])
mean((lass_pred - y[test])^2) #so far, lowest error rate

predict(reg_lass, s = reg_lass$lambda.min, type = "coefficients") #coeff of the model


#(e) Fit a PCR model on the training set, with M chosen by crossvalidation.
#Report the test error obtained, along with the value of M selected by cross-validation.
set.seed(1)
reg_pcr <- pcr(Apps ~., data = College[train, ], scale = TRUE, validation = "CV")
summary(reg_pcr)
validationplot(reg_pcr, val.type="MSEP")
#seems like the lowest CV score comes from M = 17

pcr_pred <- predict(reg_pcr, x[test, ], ncomp = 17)
mean((pcr_pred - y[test])^2) 

#(f) Fit a PLS model on the training set, with M chosen by crossvalidation.
#Report the test error obtained, along with the value of M selected by cross-validation.
set.seed(1)
reg_pls <- plsr(Apps ~., data = College[train, ], scale = TRUE, validation = "CV")
summary(reg_pls)

validationplot(reg_pls, val.type = "MSEP") #the CV's minimum is at 10

pls_preds <- predict(reg_pls, x[test, ], ncomp = 10)
mean((pls_preds - y[test] )^2)


#(g) Comment on the results obtained. How accurately can we predict the number of college 
#applications received? Is there much difference among the test errors resulting from these 
#five approaches?
mean((predict(reg, College[test, ]) - College$Apps[test])^2) #lm error
mean((ridge_pred - y[test])^2) #ridge model error
mean((lass_pred - y[test])^2) #lasso model error
mean((pcr_pred - y[test])^2) #pcr model error
mean((pls_preds - y[test] )^2) #pls model error
#By far the best performing model is the pls model
#the lm and pcr models have roughly the same error
#the ridge model performed the worst

summary(plsr(Apps ~., data=College, scale=TRUE , ncomp=10))
#the model explains 92.89% of the variance in Applications


###Question 11.#### 
###We will now try to predict per capita crime rate in the Boston data set.
library(boot)
library(glmnet)
library(glmnet)
library(pls)
library(MASS)

Boston <- MASS::Boston

#(a) Try out some of the regression methods explored in this chapter, such as best subset selection,
#the lasso, ridge regression, and PCR. Present and discuss results for the approaches that you
#consider.

#set-up
x <- model.matrix(crim ~., Boston)[ ,-1]
y <- Boston$crim


##best subset selection
#predict regsubset function
predict.regsubsets = function(object, newdata, id, ...) {
  form = as.formula(object$call[[2]])
  mat = model.matrix(form, newdata)
  coefi = coef(object, id = id)
  mat[, names(coefi)] %*% coefi
}

#k-fold c-validation (straight from the textbook)
k <- 10 #number folds
p <- ncol(Boston) - 1 #number predictors
folds <- sample(rep(1:k, length = nrow(Boston))) #sample vector of values = n folds
#length = n rows in boston
cv.errors <- matrix(NA, k, p) #value, k rows, p columns

for (i in 1:k) { #for every fold in the k-folds
  best.fit <- regsubsets(crim ~ ., data = Boston[folds != i, ], nvmax = p) #fit model on folds
  #with n var = n 
  
  for (j in 1:p) { #then for every best p-variable model 
    pred <- predict(best.fit, Boston[folds == i, ], id = j) 
    cv.errors[i, j] <- mean((Boston$crim[folds == i] - pred)^2) #find *test* error
  }
}

errors <- sqrt(apply(cv.errors, 2, mean)) #get av MSE error across all folds
which.min(errors) #min errors is for model w 12 predictors
plot(errors, pch = 19, type = "b") #plotting errors, pch is type of point used

#best model
best_model <- regsubsets(crim ~ ., data = Boston, nvmax = p)
pred_lm <- predict(best_model, Boston, id = 12)
mean((y - pred_lm)^2)

###Ridge Regression
set.seed(1) #set seed for reproduc.
reg_ridge <- cv.glmnet(x, y, alpha=0) #performs ridge reg w 10fold cv
bestlam <- reg_ridge$lambda.min #finds lambda w lowest train error, here 409 approx
ridge_pred <- predict(reg_ridge, s= bestlam, newx=x)
mean((ridge_pred - y)^2) #test error

###Lasso
set.seed(1)
reg_lass <- cv.glmnet(x, y, alpha = 1 ) #alpha = 1 -> lasso instead of ridge
lass_pred <- predict(reg_lass, s = reg_lass$lambda.min, newx = x)
mean((lass_pred - y)^2) 

###PCR 
set.seed(1)
reg_pcr <- pcr(crim ~., data = Boston, scale = TRUE, validation = "CV")
summary(reg_pcr)
validationplot(reg_pcr, val.type="MSEP") #seems like the lowest adj CV score comes from M = 8
pcr_pred <- predict(reg_pcr, x, ncomp = 8)
mean((pcr_pred - y)^2) 

###PLS 
set.seed(1)
reg_pls <- plsr(crim ~., data = Boston, scale = TRUE, validation = "CV")
summary(reg_pls)

validationplot(reg_pls, val.type = "MSEP") #the adj CV's minimum is at 10

pls_preds <- predict(reg_pls, x, ncomp = 10)
mean((pls_preds - y)^2)

#comparing test error
mean((pred_lm - y)^2)
mean((ridge_pred - y)^2) 
mean((lass_pred - y)^2) 
mean((pcr_pred - y)^2) 
mean((pls_preds - y)^2)
#very similar error rates, would pick lm, lasso or pls

#(b) Propose a model (or set of models) that seem to perform well on this data set, and justify your
#answer. Make sure that you are evaluating model performance using validation set error, 
#crossvalidation, or some other reasonable alternative, as opposed to using training error.
#the above used CV to pick models, then tested w/ training set

#the lm, lasso or pls model seems to do fairly well, though the test error rates seem fairly similar

coef(best_model, 12) #lm coeff 
predict(reg_lass, s = reg_lass$lambda.min, type= "coefficients") #lasso coeff
#lm and lasso are fairly similar
#lasso shrinks several coefficients towards 0: zn, indus, age, tax, black are all < 0.05

summary(plsr(crim ~., data = Boston, scale = TRUE, ncomp= 10))
#pls is mysterious


#(c) Does your chosen model involve all of the features in the data set? Why or why not?
#no. some of the predictors are not related to the response, hence the importance of 
#using a model which performs variable selection, e.g. lasso or BSS

###