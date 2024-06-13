library(corrgram)
library(corrplot)
library(caTools)
library(Amelia)
library(ggplot2)
library(dplyr)
library(rpart)
library(rpart.plot)
library(randomForest)
library(ISLR)
library(e1071)
library(cluster)
library(tm)
library(twitteR)
library(wordcloud)
library(RColorBrewer)
library(neuralnet)

#### getting the data 

df <- read.csv('bank_note_data.csv')

head(df)

View(df)

#### normalize the data

var(df[,3])
var(df[,4])

standard.df <- scale(df[1:4])

var(standard.df[,3])
var(standard.df[,4])

#### adding the last column back in 

new.df <- cbind(standard.df,df[5])

View(new.df)

#### train and test

sample <- sample.split(new.df,SplitRatio = 0.7)

train <- subset(new.df,sample == TRUE)
test <- subset(new.df, sample == FALSE)

#### making model

model <- neuralnet(Class ~ Image.Var + Image.Skew + Image.Curt + Entropy,train, hidden=c(5,3),
                   linear.output = FALSE)

plot(model)

predict.model <- compute(model,test[1:4])

head(predict.model$net.result)

#### rounding to 0 and 1

prediction <- sapply(predict.model$net.result,round)

head(prediction)

table(prediction)

#### confusion matrix

table(prediction,test$Class)

#### seems too good to be true as its coming out almost without any error or false positives or false negatives
#### quickly using the random forest to check the accuracy of the neural network result

df <- read.csv('bank_note_data.csv')

df$Class <- factor(df$Class)

#### EDA time

ggplot(df,aes(Image.Var)) + geom_histogram(aes(fill = Class),color='black',alpha=0.7)

#### we see from the data plotted above that the image variance really matters to confirm the bank note is real or not, which honestly is not a surprise

ggplot(df,aes(Image.Skew)) + geom_histogram(aes(fill = Class),color='black',alpha=0.7,position = 'dodge') + theme_bw()

ggplot(df,aes(Image.Var,Image.Skew)) + geom_point(aes(color=Class)) + theme_bw()

#### the scatter plot clearly shows the higher the variance or skew the higher is the probability of the bill being fake


str(df)

sample <- sample.split(df$Class,SplitRatio = 0.7)

train <- subset(df,sample == TRUE)
test <- subset(df,sample == FALSE)

model <- randomForest(Class ~.,train)

table(model$predicted)

predict.model <- predict(model,test[1:4])

table(predict.model,test$Class)

print(predict.model)

table.predict <- cbind(predict.model,test[5])

View(table.predict)

#### even the random forest is pretty accurate in this case