####Chapter 4: Classification####
##Question 10
#Produce some numerical and graphical summaries of the Weekly data. 
#Do there appear to be any patterns?
summary(Weekly)
plot(Weekly$Year, Weekly$Lag1)
plot(Weekly$Year, Weekly$Lag2)
plot(Weekly$Year, Weekly$Lag3)
plot(Weekly$Year, Weekly$Lag4)
plot(Weekly$Year, Weekly$Lag5)
#all of the lags look very similar. Increasing variance from 1995 to 2005, then around 2008.

plot(Weekly$Year, Weekly$Volume) #volume is steadily increasing over time

plot(Weekly$Year, Weekly$Direction) #direction looks evenly spaced

#(b) Use the full data set to perform a logistic regression with
#Direction as the response and the five lag variables plus Volume
#as predictors. Use the summary function to print the results. Do
#any of the predictors appear to be statistically significant? If so,
#which ones?
model <- glm(Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume, data = Weekly, family = binomial)
summary(model)
#The intercept and lag2 are significant. Lag1 has a pvalue of 0.1181, so also relatively significant

#(c) Compute the confusion matrix and overall fraction of correct
#predictions. Explain what the confusion matrix is telling you
#about the types of mistakes made by logistic regression.
glm_predict <- predict(model, data = Weekly)
glm_results <- glm_predict
glm_results[glm_results >= 0.5] <- "Up"
glm_results[glm_results < 0.5] <- "Down"
table(glm_results, Weekly$Direction)

mean(glm_results == Weekly$Direction) #46.56% correct prediction rate
sum(glm_results == "Up" &  Weekly$Direction == "Up")/sum(glm_results == "Up") #68.85 when it comes to correctly predicting upward movement
sum(glm_results == "Down" &  Weekly$Direction == "Down")/sum(glm_results == "Down") #45.23 to predict downturns

#(d) Now fit the logistic regression model using a training data period
#from 1990 to 2008, with Lag2 as the only predictor. Compute the
#confusion matrix and the overall fraction of correct predictions
#for the held out data (that is, the data from 2009 and 2010).
train <- Weekly$Year < 2009
model_train <- glm(Direction ~ Lag2, data = Weekly[train, ], family = binomial)
summary(model_train)

glm_predict <- predict(model_train, Weekly[!train,])
glm_results <- glm_predict
glm_results[glm_results >= 0.5] <- "Up"
glm_results[glm_results < 0.5] <- "Down"
table(glm_results, Weekly$Direction[!train])

mean(glm_results == Weekly$Direction[!train]) #43.23% correct prediction rate
sum(glm_results == "Up" &  Weekly$Direction[!train] == "Up")/sum(glm_results == "Up") #71.45 when it comes to correctly predicting upward movement
sum(glm_results == "Down" &  Weekly$Direction[!train] == "Down")/sum(glm_results == "Down") #42.27 to predict downturns

#(e) Repeat (d) using LDA.
lda_model <- MASS::lda(Direction ~ Lag2, data = Weekly[train, ]) 
lda_predict <- predict(lda_model, Weekly[!train, ]) 
table(lda_predict$class, Weekly$Direction[!train])

mean(lda_predict$class == Weekly$Direction[!train]) #62.5% correct prediction rate
sum(lda_predict$class == "Up" &  Weekly$Direction[!train] == "Up")/sum(lda_predict$class == "Up") #62.22 when it comes to correctly predicting upward movement
sum(lda_predict$class == "Down" &  Weekly$Direction[!train] == "Down")/sum(lda_predict$class == "Down") #64.28 to predict downturns
#LDA has a higher all-around prediction rate, and predicts upturns and downturns equally well

#(f) Repeat (d) using QDA.
qda_model <- MASS::qda(Direction ~ Lag2, data = Weekly[train, ]) 
qda_predict <- predict(qda_model, Weekly[!train, ]) 
table(qda_predict$class, Weekly$Direction[!train])

mean(qda_predict$class == Weekly$Direction[!train]) #58.65% correct prediction rate
sum(qda_predict$class == "Up" &  Weekly$Direction[!train] == "Up")/sum(qda_predict$class == "Up") #62.22 when it comes to correctly predicting upward movement
sum(qda_predict$class == "Down" &  Weekly$Direction[!train] == "Down")/sum(qda_predict$class == "Down") #64.28 to predict downturns
#qda has a lower all-around prediction rate than LDA, though it performs better than logit
#it has a problem predicting downturns however

#(g) Repeat (d) using KNN with K = 1.
train_x <- Weekly$Lag2[train] #predictor matrix, training data
test_x <- Weekly$Lag2[!train] #predictor matrix, testing data
train_y <- Weekly$Direction[train] #results vector, training data
test_y <- Weekly$Direction[!train] #results vector, testing data

set.seed(1) #set seed bc knn() breaks ties using random numbers
knn_pred <- knn(data.frame(train_x), data.frame(test_x), train_y, k = 1)
table(knn_pred, test_y)
mean(knn_pred == test_y) #predicts 50% correctly

#(h) Which of these methods appears to provide the best results on this data?
##In this case LDA seems to be giving us the best all-around performance. 

#(i) Experiment with different combinations of predictors, including
#possible transformations and interactions, for each of the
#methods. Report the variables, method, and associated confusion
#matrix that appears to provide the best results on the held
#out data. Note that you should also experiment with values for
#K in the KNN classifier.

##both glm and lda perform best with Lag2 only

###MAKING A LOOP TO FIND BEST  k
max_k <- 10
error_rate <- c(rep(0, max_k))

for (i in 1:max_k) {
  set.seed(1)
  knn_pred <- knn(data.frame(train_x), data.frame(test_x), train_y, k = i)
  
  error_rate[i] <- mean(knn_pred != test_y)
}
error_rate

#the error_rate is lowest for k = 4, though it varies depending on the seed set
knn_pred <- knn(data.frame(train_x), data.frame(test_x), train_y, k = 4)
mean(knn_pred == test_y)
#best K here has success rate of 56.73%, still not as good as LDA

##Question 11
#predict whether a given car gets high or low gas mileage based on the Auto data set.
#(a) Create a binary variable, mpg01, that contains a 1 if mpg contains
#a value above its median, and a 0 if mpg contains a value below its median. 
Auto$mpg01[Auto$mpg >= median(Auto$mpg)] <- 1
Auto$mpg01[Auto$mpg < median(Auto$mpg)] <- 0

#(b) Explore the data graphically in order to investigate the association
#between mpg01 and the other features. Which of the other
#features seem most likely to be useful in predicting mpg01? Scatterplots
#and boxplots may be useful tools to answer this question.
#Describe your findings.
cor(Auto[ , c(1:8)])
#Looking at cor() we find a strong correlation between mpg and all of the variables.
#However, cylinders, displacement and horsepower are strongly correlated between one another.
#The variables we'll be using are mpg displacement acceleration year origin

#(c) Split the data into a training set and a test set.
#we want to set aside 15% of the data for testing so
random <- c(sample(c(1:nrow(Auto)), nrow(Auto))) #randomizes Auto row number
train <- random[1:round(0.9*nrow(Auto), 0)] #selects first 90% of randomized row numbers
test <- random[round(0.9*nrow(Auto), 0):nrow(Auto)] #selects last 10% of randomized row number

#(d) Perform LDA on the training data in order to predict mpg01 using the variables that 
#seemed most associated with mpg01 in #(b). What is the test error of the model obtained?
lda_model <- MASS::lda(mpg01 ~ displacement + acceleration + year + origin, data = Auto[train, ]) 
lda_model

lda_predict <- predict(lda_model, Auto[test, ]) 
table(lda_predict$class, Auto$mpg01[test])

mean(lda_predict$class == Auto$mpg01[test]) #85% correct prediction rate
sum(lda_predict$class == 1 &  Auto$mpg01[test] == 1)/sum(lda_predict$class == 1) #75 correct for above median
sum(lda_predict$class == 0 &  Auto$mpg01[test] == 0)/sum(lda_predict$class == 0) #100 to predict below median

#(e) Perform QDA on the training data in order to predict mpg01 using the variables that seemed 
#most associated with mpg01 in (b). What is the test error of the model obtained?
qda_model <- MASS::qda(mpg01 ~ displacement + acceleration + year + origin, data = Auto[train, ]) 
qda_model

qda_predict <- predict(qda_model, Auto[test, ]) 
table(qda_predict$class, Auto$mpg01[test])

mean(qda_predict$class == Auto$mpg01[test]) #85% correct prediction rate
sum(qda_predict$class == 1 &  Auto$mpg01[test] == 1)/sum(qda_predict$class == 1) #75 correct for above median
sum(qda_predict$class == 0 &  Auto$mpg01[test] == 0)/sum(qda_predict$class == 0) #100 to predict below median

#the same as for LDA

#(f) Perform logistic regression on the training data in order to predict
#mpg01 using the variables that seemed most associated with
#mpg01 in (b). What is the test error of the model obtained?
glm_model <- glm(mpg01 ~ displacement + acceleration + year + origin, data = Auto[train, ])
summary(glm_model)

glm_predict <- predict(glm_model, Auto[test,])
glm_results <- glm_predict
glm_results[glm_results >= 0.5] <- 1
glm_results[glm_results < 0.5] <- 0
table(glm_results, Auto$mpg01[test])

mean(glm_results == Auto$mpg01[test]) #85% correct prediction rate

#(g) Perform KNN on the training data, with several values of K, in
#order to predict mpg01. Use only the variables that seemed most
#associated with mpg01 in (b). What test errors do you obtain?
#Which value of K seems to perform the best on this data set?
train_x <- Auto[train, c(3, 6, 7, 8)] #predictor matrix, training data
test_x <- Auto[test, c(3, 6, 7, 8)] #predictor matrix, testing data
train_y <- Auto$mpg01[train] #results vector, training data
test_y <- Auto$mpg01[test] #results vector, testing data

##ALT is use cbind(colnames)[train, ]

set.seed(1) #set seed bc knn() breaks ties using random numbers
knn_pred <- knn(train_x, test_x, train_y, k = 1)
table(knn_pred, test_y)
mean(knn_pred == test_y) #predicts 90% correctly

#loop to find best k
max_k <- 10
error_rate <- c(rep(0, max_k))

for (i in 1:max_k) {
  set.seed(4656)
  knn_pred <- knn(train_x, test_x, train_y, k = i)
  error_rate[i] <- mean(knn_pred != test_y)
}
error_rate

#k = 1 is best

##Exercise 13
#Using the Boston data set, fit classification models in order to predict
#whether a given suburb has a crime rate above or below the median.
#Explore logistic regression, LDA, and KNN models using various subsets
#of the predictors. Describe your findings.
Boston <- MASS::Boston
cor(Boston)
#after looking around, will use everything but chas
#high corr between indus>nox/age, dis>zn/age, rad>nox, lstad>medv
#keeping variables with highest corr w crime
#indus chas rm dis rad tax ptratio black lstat

#convert crime rate > median into a probability
Boston$crim_prob <- Boston$crim
Boston$crim_prob[Boston$crim_prob >= median(Boston$crim_prob)] <- 1
Boston$crim_prob[Boston$crim_prob < median(Boston$crim_prob)] <- 0

#need to create training data
random <- sample(c(1:nrow(Boston)), nrow(Boston))
train <- sample(c(1:nrow(Boston)), nrow(Boston))[1:round(nrow(Boston)*0.9, 0)]
test <- sample(c(1:nrow(Boston)), nrow(Boston))[round(nrow(Boston)*0.9, 0):nrow(Boston)]
model_train <- glm(crim_prob ~ indus + chas + rm + dis + rad + tax + ptratio + black + lstat, data = Boston[train, ], family = binomial)
summary(model_train)
#all of the variables but chas are relatively significant

glm_predict <- predict(model_train, Boston[test,])
glm_results <- glm_predict
glm_results[glm_results >= 0.5] <- 1
glm_results[glm_results < 0.5] <- 0
table(glm_results, Boston$crim_prob[test])

mean(glm_results == Boston$crim_prob[test]) #82.69% correct prediction rate
sum(glm_results == 1 &  Boston$crim_prob[test] == 1)/sum(glm_results == 1) #100% when it comes to correctly predicting above
sum(glm_results == 0 &  Boston$crim_prob[test] == 0)/sum(glm_results == 0) #65.38 to predict below

#excluding chas
model_train <- glm(crim_prob ~ indus + rm + dis + rad + tax + ptratio + black + lstat, data = Boston[train, ], family = binomial)
summary(model_train)

glm_predict <- predict(model_train, Boston[test,])
glm_results <- glm_predict
glm_results[glm_results >= 0.5] <- 1
glm_results[glm_results < 0.5] <- 0
table(glm_results, Boston$crim_prob[test])

mean(glm_results == Boston$crim_prob[test]) #82.69% correct prediction rate
sum(glm_results == 1 &  Boston$crim_prob[test] == 1)/sum(glm_results == 1) #100% when it comes to correctly predicting above
sum(glm_results == 0 &  Boston$crim_prob[test] == 0)/sum(glm_results == 0) #65.38 to predict below
#doesnt change much but makes for a more concise model

#trying out LDA
lda_model <- MASS::lda(crim_prob ~ indus + rm + dis + rad + tax + ptratio + black + lstat, data = Boston[train, ]) 
lda_model

lda_predict <- predict(lda_model, Boston[test, ]) 
table(lda_predict$class, Boston$crim_prob[test])

mean(lda_predict$class == Boston$crim_prob[test]) #84.6% correct prediction rate
sum(lda_predict$class == 1 &  Boston$crim_prob[test] == 1)/sum(lda_predict$class == 1) #100 correct for above median
sum(lda_predict$class == 0 &  Boston$crim_prob[test] == 0)/sum(lda_predict$class == 0) #68 to predict below median

#QDA
qda_model <- MASS::qda(crim_prob ~ indus + rm + dis + rad + tax + ptratio + black + lstat, data = Boston[train, ]) 
qda_model

qda_predict <- predict(qda_model, Boston[test, ]) 
table(qda_predict$class, Boston$crim_prob[test])

mean(qda_predict$class == Boston$crim_prob[test]) #80.7% correct prediction rate
sum(qda_predict$class == 1 &  Boston$crim_prob[test] == 1)/sum(qda_predict$class == 1) #100 correct for above median
sum(qda_predict$class == 0 &  Boston$crim_prob[test] == 0)/sum(qda_predict$class == 0) #63 to predict below median

#KNN
train_x <- cbind(Boston$indus, Boston$rm, Boston$dis, Boston$rad, Boston$tax, Boston$ptratio, Boston$black, Boston$lstat)[train, ] #predictor matrix, training data
test_x <- cbind(Boston$indus, Boston$rm, Boston$dis, Boston$rad, Boston$tax, Boston$ptratio, Boston$black, Boston$lstat)[test, ] #predictor matrix, testing data
train_y <- Boston$crim_prob[train] #results vector, training data
test_y <- Boston$crim_prob[test] #results vector, testing data

set.seed(1) #set seed bc knn() breaks ties using random numbers
knn_pred <- knn(train_x, test_x, train_y, k = 1)
table(knn_pred, test_y)
mean(knn_pred == test_y) #predicts 100% correctly

#loop to find best k
max_k <- 10
error_rate <- c(rep(0, max_k))

for (i in 1:max_k) {
  set.seed(4746)
  knn_pred <- knn(train_x, test_x, train_y, k = i)
  error_rate[i] <- mean(knn_pred != test_y)
}
error_rate

#knn with k = 1 has, BY FAR, the best performance. I checked bc feeling skeptical but it has a 
#100% prediction rate