install.packages("DMwR") 
install.packages('caret', dependencies = TRUE)
install.packages("e1071") 
install.packages("corrplot")
library(ggplot2)
library(dplyr)
library(randomForest)
library(DMwR2)
library("DMwR")
library("caret")
library(e1071) 
library(rpart.plot)
library(reshape)
library(scales)
library(corrplot)

getwd()
setwd("/Users/yangliu/Documents/Fall 2018/INFO 523/final project")
data <- read.csv("aldataset.csv")
summary(data) 
dim(data) #check data

#handle missing value
missing.value.rows <- filter(data, !complete.cases(data)) 
missing.value.rows #only 11 out of 7043 rows contains missing values
View(missing.value.rows) #the missing values is only shown in the total charges column
max(apply(data, 1, function(x) sum(is.na(x)))) #max missing value in a row is 1
sort(apply(data, 2, function(x) sum(is.na(x)))) #the missing values is only shown in the total charges column
sapply(data,class) #check the data type fo columns
data2 <- na.omit(data)
cor(data2$MonthlyCharges, data2$TotalCharges) #not quite corelated, not using regression method to fill in NA
data2 <- data2 %>%
  mutate(new_total = MonthlyCharges*tenure) 
cor(data2$new_total, data2$TotalCharges) #the correlation of total charges and monthly charge*tenure is 0.9995
data3 <- filter(data2, data2$tenure==0)
dim(data3) #after omit NA, there is no tenure of 0, which means the total charge of tenure of 0 is NA. which means that we can fill the NA with 0.
data <- na.omit(data) #we finally decided to remove the 11 records with missing values

# #filling in NA using mean, median or something else?
# install.packages("car")
# library(car)
# qqPlot(data$TotalCharges, main="Normal QQ plot of TotalCharges")
# #the straight line doesn't fit into the data well, so the mean is not a representative value for TotalCharges, we will use median to fill the NA
# #replace missing value with median
# data[is.na(data$TotalCharges), "TotalCharges"] <- median(data$TotalCharges, na.rm=TRUE)
# summary(data) #check the data again, now we don't have missing values

#outlier detection
#we used boxplot to detect outlier, the result shows that there are no outlier
ggplot(data, aes(x = "MonthlyCharges", y = MonthlyCharges)) + geom_boxplot()
ggplot(data, aes(x = "TotalCharges", y = MonthlyCharges)) + geom_boxplot()
ggplot(data, aes(x = "tenure", y = tenure)) + geom_boxplot()
#to double check, we also used histogram to detect outlier, the result shows that there are no outlier.
#Histogram of MonthlyCharges
ggplot(data = data, mapping = aes(x = MonthlyCharges)) +
  geom_histogram(bins = 20, fill = "black") +
  labs(x = "MonthlyCharges") +
  ggtitle("Histogram of MonthlyCharges") +
  theme(plot.title = element_text(hjust = 0.5))

#Histogram of TotalCharges
ggplot(data = data, mapping = aes(x = TotalCharges)) +
  geom_histogram(bins = 20, fill = "black") +
  labs(x = "TotalCharges") +
  ggtitle("Histogram of TotalCharges") +
  theme(plot.title = element_text(hjust = 0.5))

#Histogram of tenure
ggplot(data = data, mapping = aes(x = tenure)) +
  geom_histogram(bins = 20, fill = "black") +
  labs(x = "tenures") +
  ggtitle("Histogram of tenure") +
  theme(plot.title = element_text(hjust = 0.5))
# data discretization
#Tenure
summary(data$tenure)
data$newtenure <- cut(data$tenure,5,labels=c("1-15", "15-29", "29-44", "44-58", "58-72")) #add labels
table(data$newtenure)
head(data)
summary(data$newtenure)

#MonthlyCharges
summary(data$MonthlyCharges)
data$newMonthlyCharges <- cut(data$MonthlyCharges,5, labels=c("18.1-38.4", "38.4-58.5", "58.5-78.6", "78.6-98.7", "98.7-119")) #add labels
table(data$newMonthlyCharges)
head(data)
summary(data$newMonthlyCharges)

#TotalCharges
summary(data$TotalCharges)
data$newTotalCharges <- cut(data$TotalCharges,5, labels=c("very low", "low", "medium", "high", "very high")) #add labels
table(data$newTotalCharges)
head(data)
summary(data$newTotalCharges)


#descriptive data analysis
#data visualization
# visualize total number of customer churn and not churn
summary(data$Churn)
ggplot(data = data, aes(x = Churn, y = ..count..)) +
  geom_bar(width=0.5)
#check whether customer churn is related to gender
colnames(data)
ggplot(data=data, aes(x = gender, fill = Churn)) + 
  geom_bar(width=0.5) + scale_fill_brewer(direction = -1)
#check whether customer churn is related to contract
ggplot(data=data, aes(x = Contract, fill = Churn)) + 
  geom_bar(width=0.5) + scale_fill_brewer(direction = -1) 
#check whether customer churn is related to internet service
ggplot(data=data, aes(x = InternetService, fill = Churn)) + 
  geom_bar(width=0.5) + scale_fill_brewer(direction = -1)
#check whether customer churn is related to tenure
ggplot(data=data, aes(x = newtenure, fill = Churn)) + 
  geom_bar(width=0.5) + scale_fill_brewer(direction = -1)
#check whether customer churn is related to monthly charge
ggplot(data=data, aes(x = newMonthlyCharges, fill = Churn)) + 
  geom_bar(width=0.5) + scale_fill_brewer(direction = -1)

#predictive data analysis
#data modeling
colnames(data)
selectData <- data[, !(colnames(data) %in% c("customerID", "MonthlyCharges","TotalCharges", "tenure"))]
churn<- filter(selectData, Churn=='Yes')
dim(churn)
nochurn <- filter(selectData, Churn=='No')
rndSample <- sample(1:nrow(nochurn),nrow(churn))
nochurn2 <- nochurn[rndSample, ]
dim(nochurn2)
cdata <- rbind(churn, nochurn2)
dim(cdata)
cdata$SeniorCitizen <- as.factor(cdata$SeniorCitizen)
cdata <- cdata[sample(nrow(cdata)),]

train<-sample_frac(cdata, 0.7)
sid<-as.numeric(rownames(train)) # because rownames() returns character
test<-cdata[-sid,]
dim(train)
dim(test)

#decision tree
tree <- rpartXse(Churn~ ., train, se=0.5)
#decision tree visualization
prp(tree, type=0, extra=101, roundint = FALSE)
prediction <- predict(tree, test, type="class")
result <- (cm <- table(prediction, test$Churn))
result
accuracy <- (result[1,1]+result[2,2])/(result[1,1]+result[1,2]+result[2,2]+result[2,1]) 
accuracy
precision <- result[2,2]/(result[2,2]+result[2,1])
precision
recall <- result[2,2]/(result[2,2]+result[1,2])
recall
F1 <- 2*precision*recall/(precision+recall)
F1

#random forest
rfm <- randomForest(Churn ~ ., train, ntree=750)
#predict test data 
rfpred <- predict(rfm, test, type="class")
result<- table(rfpred, test$Churn)
result
accuracy <- (result[1,1]+result[2,2])/(result[1,1]+result[1,2]+result[2,2]+result[2,1]) 
accuracy
precision <- result[2,2]/(result[2,2]+result[2,1])
precision
recall <- result[2,2]/(result[2,2]+result[1,2])
recall
F1 <- 2*precision*recall/(precision+recall)
F1

#SVM
svm_Linear <- svm(Churn ~., data = train, kernel = "linear")
test_pred <- predict(svm_Linear, test)
result <- table(test_pred, test$Churn)
result
accuracy <- (result[1,1]+result[2,2])/(result[1,1]+result[1,2]+result[2,2]+result[2,1]) 
accuracy
precision <- result[2,2]/(result[2,2]+result[2,1])
precision
recall <- result[2,2]/(result[2,2]+result[1,2])
recall
F1 <- 2*precision*recall/(precision+recall)
F1

#Naive Bayes
Naivebayes<- naiveBayes(Churn ~., data = train)
test_pred <- predict(svm_Linear, newdata = test)
result <- table(test_pred, test$Churn)
result
accuracy <- (result[1,1]+result[2,2])/(result[1,1]+result[1,2]+result[2,2]+result[2,1]) 
accuracy
precision <- result[2,2]/(result[2,2]+result[2,1])
precision
recall <- result[2,2]/(result[2,2]+result[1,2])
recall
F1 <- 2*precision*recall/(precision+recall)
F1

#comparing results using visualization
model <- c("Decision Tree", "Random Forest", "SVM", "Naive Bayes")
accuracy <- c(0.781, 0.884, 0.756, 0.760)
precision <- c(0.787, 0.882, 0.733, 0.724)
recall <- c(0.775, 0.888, 0.805, 0.843)
F1 <- c(0.781, 0.885, 0.767, 0.779)
df <- data.frame(model, accuracy, precision, recall, F1)
# melt the data frame for plotting
df.m <- melt(df, id.vars='model')
# plot all evaluation results
ggplot(df.m, aes(x=model, y=value)) +   
  geom_bar(aes(fill = variable), position = "dodge", stat="identity") + scale_fill_brewer() 
write.csv(df, file = "MyData.csv",row.names=FALSE)

# # SVM cross validation
# trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
# set.seed(3233)
# 
# svm_Linear <- train(Churn ~., data = cdata, method = "svmLinear",
#                     trControl=trctrl,
#                     preProcess = c("center", "scale"),
#                     tuneLength = 12)
# svm_Linear
# result <- confusionMatrix(svm_Linear)
# result
# result1 <- result$table
# precision <- result1[2,2]/(result1[2,2]+result1[2,1])
# precision
# recall <- result1[2,2]/(result1[2,2]+result1[1,2])
# recall
# F1 <- 2*precision*recall/(precision+recall)
# F1

#random forest gives the best performance, so we decided to use random forest as our prediction model, so we tuning parameter for random forest
#Parameter tunning for random forest
tree_number <- c(50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000)
recall_value <- c()
F1_score <- c()
for(i in tree_number) {
  rfm <- randomForest(Churn ~ ., train, ntree=i)
  rfpred <- predict(rfm, test, type="class")
  result<- table(rfpred, test$Churn)
  precision <- result[2,2]/(result[2,2]+result[2,1])
  recall <- result[2,2]/(result[2,2]+result[1,2])
  F1 <- 2*precision*recall/(precision+recall)
  recall_value <- append(recall_value, recall, after=length(recall_value))
  F1_score <- append(F1_score, F1, after=length(F1_score))
}
recall_value
F1_score

df2 <- data.frame(tree_number, F1_score)
ggplot(data = df2, aes(x = tree_number, y = F1_score)) +
  geom_point() +
  geom_line()
# the result shows that the model gives best performance with 750 trees

#Check performance and feature importance
rfm <- randomForest(Churn ~ ., train, ntree=750)
rfpred <- predict(rfm, test, type="class")
result<- table(rfpred, test$Churn)
accuracy <- (result[1,1]+result[2,2])/(result[1,1]+result[1,2]+result[2,2]+result[2,1]) 
precision <- result[2,2]/(result[2,2]+result[2,1])
recall <- result[2,2]/(result[2,2]+result[1,2])
F1 <- 2*precision*recall/(precision+recall)
accuracy
precision
recall
F1
importance(rfm)        
varImpPlot(rfm) 

#run 10 fold cross validation for random forest with ntree 750 to see if it will improve the performance
#Randomly shuffle the data
mydata<-cdata[sample(nrow(cdata)),]
#Create 10 equally size folds
folds <- cut(seq(1,nrow(mydata)),breaks=10,labels=FALSE)
#Perform 10 fold cross validation
accuracy_value <- c()
precision_value <- c()
recall_value2 <- c()
F1_score2 <- c()
for(i in 1:10){
  #Segement data by fold using the which() function 
  testIndexes <- which(folds==i,arr.ind=TRUE)
  test2 <- mydata[testIndexes, ]
  train2 <- mydata[-testIndexes, ]
  #Use the test and train data for random forest model
  rfm <- randomForest(Churn ~ ., train2, ntree=750)
  rfpred <- predict(rfm, test2, type="class")
  result<- table(rfpred, test2$Churn)
  accuracy <- (result[1,1]+result[2,2])/(result[1,1]+result[1,2]+result[2,2]+result[2,1]) 
  precision <- result[2,2]/(result[2,2]+result[2,1])
  recall <- result[2,2]/(result[2,2]+result[1,2])
  F1 <- 2*precision*recall/(precision+recall)
  accuracy_value <- append(accuracy_value, accuracy, after=length(accuracy_value))
  precision_value <- append(precision_value, precision, after=length(precision_value))
  recall_value2 <- append(recall_value2, recall, after=length(recall_value2))
  F1_score2 <- append(F1_score2, F1, after=length(F1_score2))
}
accuracy <- mean(accuracy_value)
accuracy
precision <- mean(precision_value)
precision
recall <- mean(recalls)
recall
F1_score <- mean(F1_scores)
F1_score



