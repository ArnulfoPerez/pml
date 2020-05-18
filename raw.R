library(caret)

library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)

library(randomForest)

library(corrplot)

library(gbm)

if (!file.exists("./data")) {
  
  dir.create("./data")
  
}

trainUrl <-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"

testUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

trainFile <- "./data/pml-training.csv"

testFile  <- "./data/pml-testing.csv"

if (!file.exists(trainFile)) {
  
  download.file(trainUrl, destfile=trainFile, method="curl")
  
}

if (!file.exists(testFile)) {
  
  download.file(testUrl, destfile=testFile, method="curl")
  
}

train_in <- read.csv(trainFile)

valid_in <- read.csv(testFile)

#train_in <- read.csv('./pml-training.csv', header=T)
#valid_in <- read.csv('./pml-testing.csv', header=T)
dim(train_in)

trainData<- train_in[, colSums(is.na(train_in)) == 0]
validData <- valid_in[, colSums(is.na(valid_in)) == 0]
dim(trainData)

dim(validData)

trainData <- trainData[, -c(1:7)]
validData <- validData[, -c(1:7)]
dim(trainData)

dim(validData)

set.seed(1234) 
inTrain <- createDataPartition(trainData$classe, p = 0.7, list = FALSE)
trainData <- trainData[inTrain, ]
testData <- trainData[-inTrain, ]
dim(trainData)

dim(testData)

NZV <- nearZeroVar(trainData)
trainData <- trainData[, -NZV]
testData  <- testData[, -NZV]
dim(trainData)

dim(testData)

cor_mat <- cor(trainData[, -53])
corrplot(cor_mat, order = "FPC", method = "color", type = "upper", 
         tl.cex = 0.8, tl.col = rgb(0, 0, 0))

highlyCorrelated = findCorrelation(cor_mat, cutoff=0.75)
names(trainData)[highlyCorrelated]

set.seed(12345)


controlRF <- trainControl(method="cv", number=3, verboseIter=FALSE)
modRF1 <- train(classe ~ ., data=trainData, method="rf", trControl=controlRF)
modRF1$finalModel

predictRF1 <- predict(modRF1, newdata=testData)
cmrf <- confusionMatrix(predictRF1, factor(testData$classe))
cmrf

plot(modRF1)

plot(cmrf$table, col = cmrf$byClass, 
     main = paste("Random Forest Confusion Matrix: Accuracy =", 
                  round(cmrf$overall['Accuracy'], 4)))



Results <- predict(modRF1, newdata=validData)
Results
