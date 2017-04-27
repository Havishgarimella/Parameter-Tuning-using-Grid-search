library(caret)
library(e1071)
library(randomForest)
install.packages("adabag")
library(adabag)
install.packages("pROC")
library(pROC)

#set directory
setwd("C:\\Users\\hvg15\\Desktop\\R\\lab (1)")

#input the datafile
lab <- read.csv("dataset.csv", header = FALSE, sep=" ")

#look into data
View(head(lab))

str(lab)

#target variable plot
hist(lab$V1, col="grey")
table(lab$V1)

colnames(lab) <- c("target","var1","var2","var3","var4","var5","var6","var7","var8","var9","var10","var11","var12","var13","var14","var15","var16","var17")

#plots of all variables
pdf(file="initial-plots.pdf")


for (i in 2:18) 
  hist(lab[,i], xlab=colnames(lab)[i], data = lab)

dev.off()

#deleting var16 and var17 columns
lab$var16 <- NULL
lab$var17 <- NULL

#data manupulation of skewed columns
#logarthmic transformations of var1, var2, var3, var4, var5, var6, var12, var13, var14
lab$var12 <- as.numeric(lab$var12)
lab$var14 <- as.numeric(lab$var14)
lab$target <- as.factor(lab$target)
#lab$var1 <- log10(lab$var1)
#lab$var2 <- log10(lab$var2)
#lab$var3 <- log10(lab$var3)
#lab$var4 <- log10(lab$var4)
#lab$var5 <- log10(lab$var5)
#lab$var6 <- log10(lab$var6)
#lab$var12 <- log10(lab$var12)
#lab$var13 <- log10(lab$var13)
#lab$var14 <- log10(lab$var14)
#log_columns <- c("var1","var2","var3","var4","var5","var6","var12","var13","var14")
#lab[,log_columns] <- sapply(lab, function(x) log10(x))


####Grid search for tuning Hyper Parameters


#naive bayes
x <- lab[,-1]
y <- as.factor(lab$target)
model_nb <- train(x,y,'nb',trControl = trainControl(method = 'cv', number = 10))
model_nb


# Create model with default paramters for RandomForest
control <- trainControl(method="repeatedcv", number=10, repeats=3)
seed <- 7
metric <- "Accuracy"
set.seed(seed)
mtry <- sqrt(ncol(lab))
tunegrid <- expand.grid(.mtry=mtry)
rf_default <- train(target~., data=lab, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_default)

#adaboost
Grid <- expand.grid(maxdepth=c(25,30),nu=2,iter=c(100,120))
results_ada = train(target~., data=lab, method="ada",
                    trControl=trainControl(method = 'cv', number = 10),tuneGrid=Grid)



#svm
tuneResult <- tune(svm,  target~.,  data = lab,
                   ranges = list(epsilon = seq(0,1,0.1), cost = 2^(2:9))
)

print(tuneResult)
# best performance:  epsilon 0 cost 32
# Draw the tuning graph
plot(tuneResult)




###
### Let's do a 10-fold cross validation.
###

###
### First let's see how we get the dimensions of our data.
###

dim(lab)
dim(lab)[1]

###
### At random assign values 1 to 10 to each row.  10 comes from our "10"-fold cv.
###

no.of.folds = 10
no.of.folds 

###
### At random pick each row to be in one of the 10 folds. Note that the fold may not 
### contain equal number of rows.  This is okay as long as we have a decent amount of data.
###
### The function set.seed() sets the random number generator to a starting value.
### This will ensure that we will all get the same results.
###


set.seed(778899)

index.values = sample(1:no.of.folds, size = dim(lab)[1], replace = TRUE)
head(index.values)
table(index.values)/dim(lab)[1]

###
### Say we want to see which rows are in fold 1.  Here's how you could do it.
### Note the double equal sign: ==
###

which(index.values == 1)

###
### The vector test.mse is going to contain the k mse values.  NA is an R special character for missing value.
### NA = Not Available
###

precision_naivebayes = rep(0, no.of.folds)
precision_naivebayes

recall_naivebayes = rep(0,no.of.folds)
recall_naivebayes

F1_naivebayes = rep(0,no.of.folds)
F1_naivebayes



for (i in 1:no.of.folds)
{
  index.out            = which(index.values == i)                             ### These are the indices of the rows that will be left out.
  left.out.data        = lab[  index.out, ]                                  ### This subset of the data is left out. (about 1/10)
  left.in.data         = lab[ -index.out, ]                                  ### This subset of the data is used to get our model. (about 9/10)
  tmp.nb               = naiveBayes(target ~ ., data = left.in.data)                 ### Perform naivebayes using the data that is left in.
  tmp.predicted.values = predict(tmp.nb, newdata = left.out.data)             ### Predict the y values for the data that was left out
  precision_naivebayes[i]          = posPredValue(tmp.predicted.values, left.out.data[,1])### Get precision
  recall_naivebayes[i]             = sensitivity(tmp.predicted.values,left.out.data[,1])  ###Get Recall
  F1_naivebayes[i]                 = (2 * precision_naivebayes[i] * recall_naivebayes[i]) / (precision_naivebayes[i] + recall_naivebayes[i])###Get F-1
}
mean(precision_naivebayes)
mean(recall_naivebayes)
mean(F1_naivebayes)


#Randomforest model
precision_randomForest = rep(0, no.of.folds)
precision_randomForest

recall_randomForest = rep(0,no.of.folds)
recall_randomForest

F1_randomForest = rep(0,no.of.folds)
F1_randomForest

for (i in 1:no.of.folds)
{
  index.out            = which(index.values == i)                             ### These are the indices of the rows that will be left out.
  left.out.data        = lab[  index.out, ]                                  ### This subset of the data is left out. (about 1/10)
  left.in.data         = lab[ -index.out, ]                                  ### This subset of the data is used to get our model. (about 9/10)
  tmp.rf               = randomForest(target ~ ., data = left.in.data)                 ### Perform randomForest using the data that is left in.
  tmp.predicted.values = predict(tmp.rf, newdata = left.out.data)             ### Predict the y values for the data that was left out
  precision_randomForest[i]          = posPredValue(tmp.predicted.values, left.out.data[,1])### Get precision
  recall_randomForest[i]             = sensitivity(tmp.predicted.values,left.out.data[,1])  ###Get Recall
  F1_randomForest[i]                 = (2 * precision_randomForest[i] * recall_randomForest[i]) / (precision_randomForest[i] + recall_randomForest[i])###Get F-1
}

mean(precision_randomForest)
mean(recall_randomForest)
mean(F1_randomForest)


#Adaboost model 2nd iteration
precision_adaboost_1 = rep(0, no.of.folds)
precision_adaboost_1

recall_adaboost_1 = rep(0,no.of.folds)
recall_adaboost_1

F1_adaboost_1 = rep(0,no.of.folds)
F1_adaboost_1

for (i in 1:no.of.folds)
{
  index.out            = which(index.values == i)                             ### These are the indices of the rows that will be left out.
  left.out.data        = lab[  index.out, ]                                  ### This subset of the data is left out. (about 1/10)
  left.in.data         = lab[ -index.out, ]                                  ### This subset of the data is used to get our model. (about 9/10)
  tmp.ab               = boosting(target ~ ., data = left.in.data, boos = TRUE, mfinal = 120,coeflearn = "Breiman")                 ### Perform randomForest using the data that is left in.
  tmp.predicted.values = predict(tmp.ab, newdata = left.out.data)             ### Predict the y values for the data that was left out
  tmp.predicted.values_ada = as.factor(tmp.predicted.values$class)
  precision_adaboost_1[i]          = posPredValue(tmp.predicted.values_ada, left.out.data[,1])### Get precision
  recall_adaboost_1[i]             = sensitivity(tmp.predicted.values_ada,left.out.data[,1])  ###Get Recall
  F1_adaboost_1[i]                 = (2 * precision_adaboost_1[i] * recall_adaboost_1[i]) / (precision_adaboost_1[i] + recall_adaboost_1[i])###Get F-1
}

mean(precision_adaboost_1)
mean(recall_adaboost_1)
mean(F1_adaboost_1)

#SVM model
precision_svm = rep(0, no.of.folds)
precision_svm

recall_svm = rep(0,no.of.folds)
recall_svm

F1_svm = rep(0,no.of.folds)
F1_svm

for (i in 1:no.of.folds)
{
  index.out            = which(index.values == i)                             ### These are the indices of the rows that will be left out.
  left.out.data        = lab[  index.out, ]                                  ### This subset of the data is left out. (about 1/10)
  left.in.data         = lab[ -index.out, ]                                  ### This subset of the data is used to get our model. (about 9/10)
  tmp.svm               = svm(target ~ ., data = left.in.data, cost = 32, epsilon = 0)                 ### Perform randomForest using the data that is left in.
  tmp.predicted.values = predict(tmp.svm, newdata = left.out.data)             ### Predict the y values for the data that was left out
  precision_svm[i]          = posPredValue(tmp.predicted.values, left.out.data[,1])### Get precision
  recall_svm[i]             = sensitivity(tmp.predicted.values,left.out.data[,1])  ###Get Recall
  F1_svm[i]                 = (2 * precision_svm[i] * recall_svm[i]) / (precision_svm[i] + recall_svm[i])###Get F-1
}

mean(precision_svm)
mean(recall_svm)
mean(F1_svm)




