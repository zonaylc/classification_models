#### Load packages
library(knitr)
library(dplyr)
library(ggcorrplot)
library(caret)
library(rpart)
library(skimr)
library(partykit)
library(randomForest)
library(ipred)
library(MASS)
library(pROC)
library(expss)

## 1 Loading Data
df <- read.csv('CloudPredict2021.csv')

## 2 Binary Classification
### 2.1 Data Exploration
#The number of the cloudy days in the dataset is 29418; clear day is 69555. And the proportion of the cloudy days to clear days is 0.423. The percentage of the cloudy days is 0.3 and clear days is 0.7 respectively in the dataset. So the samples of the classes are imbalanced.  
counts <- as.data.frame(table(df$CloudTomorrow))
print(paste("Proportion of cloudy to clear:", counts$Freq[2] / counts$Freq[1]))
print(paste("Percentage of cloudy days:", counts$Freq[2] / (counts$Freq[1]+counts$Freq[2])))
print(paste("Percentage of clear days:", counts$Freq[1] / (counts$Freq[1]+counts$Freq[2])))

#The description of the real valued predictors:
skim(df) #(df[,c(1:9, 14:21)])

#Data summary after omitting the missing values

## Omit the missing values
df <- na.omit(df)
# Check the data change
skim(df)

counts2 <- as.data.frame(table(df$CloudTomorrow))
print(paste("Proportion of cloudy to clear:", counts2$Freq[2] / counts2$Freq[1]))
print(paste("Percentage of cloudy days:", counts2$Freq[2] / (counts2$Freq[1]+counts2$Freq[2])))
print(paste("Percentage of clear days:", counts2$Freq[1] / (counts2$Freq[1]+counts2$Freq[2])))

#Obersvering the data distribtions and the descriptions of the real valued predictors after ommiting the missing values, it's surprising that the proportion of the target varibale becomes much more even, and now the class distribution is balanced. However, this could also cause another  problem, beacuse the data cannot truly reflect the real world and make the classification a huge bias to the future data. In the meanwhile, we can observe that the year distrubution is not balanced for every year, and it gets a little bit worth after omitting the missing value. A lot of missing values might come from the same year's observations. And it can also make the influence on the target class distribution.
#Another worth to mention is that the predictor Rainfall has very few extreme values, and it makes the standard deviation to be a value with no sense. Evaporation doesn't have so a unrational standard deviation, but it also have the same problem as Rainfall. These two variables would always get the similar values excluding the outliers, which means these variables are not informative but could be misleading to the classifier. Therefore, we can remove Rainfall and Evaporation for our analysis.
#Next step, we can always check the correlation between predictors and have a double check to not use the similar but not predictive variables.
ggcorrplot(cor(as.matrix(df[,c(1:9, 14:21, 23)])),title='Correlation between numeric variables',tl.srt = 90)

#Through the correlation heat map, it verifies that the high tempture caused lower pressure. And tempture can dominate several predictor's value, but we cannot judge which predictors are more important than other variables without further analysis.


### 2.2 Pre-processing for classification
#1. Convert the categorical data with one-hot encoding.
#2. Convert (Yes, No) into (1,0)
#3. Convert the target variable into factor for classification.


# Deal with categorical data
dmy <- dummyVars(~ WindDir3pm + WindDir9am + WindGustDir, data = df)
onehot <- data.frame(predict(dmy, newdata = df))

df$RainToday <- ifelse(df$RainToday=="Yes",1,0)
clf_df <- cbind(df[,c(1:6,9, 14:21, 23)], onehot)
clf_df$CloudTomorrow <- as.factor(clf_df$CloudTomorrow)
head(clf_df, 3)


### 2.3 Train-test Split
set.seed(12345)
default_idx = sample(nrow(clf_df)*0.7) #sample the index
train =  clf_df[default_idx, ]
test = clf_df[-default_idx, ]


### 2.4
#Fit a decision tree with training data.
set.seed(1)
dt = rpart(CloudTomorrow~., data = train)
plot(as.party(dt))


### 2.5
#Fit a logistic regression classifier with training data.
model_glm = glm(CloudTomorrow~., data = train, family = "binomial")
model_glm


### 2.6
#Random Forest - set the parameter mtry=8 since we have 63 features after onehot encoding, and the mtry number is squared root of number of features.
rf_mod = randomForest(CloudTomorrow~., data = train, ntree=150, mtry = 8)
print(paste("Training Accuracy:", sum(diag(rf_mod$confusion))/nrow(train)))
varImpPlot(rf_mod)


### 2.7
##Bagged Tree
bag <- bagging(CloudTomorrow ~., data = train, nbagg = 50, coob = TRUE, control = rpart.control(minsplit = 2, cp = 0))
bag

### 2.8
##LDA
# Only use the most important predictors from the given result of random forest to avoid unecessary collinearaity
clf_lda = lda(CloudTomorrow ~ Sunshine + Humidity3pm + Location + Humidity9am + Pressure9am + Pressure3pm + MinTemp + MaxTemp + Day + Year + Month, data = train)
clf_lda


### 2.9
##QDA
# Use the same predictors for qda 
clf_qda = qda(CloudTomorrow ~ Sunshine + Humidity3pm + Location + Humidity9am + Pressure9am + Pressure3pm + MinTemp + MaxTemp + Day + Year + Month, data = train)

clf_qda

## 3
### 3.1
#The confusion matrix and the accuracy rate of the test data.
# Prepare accuracy function
calc_class_acc = function(actual, predicted) {
  mean(actual == predicted)
}


#1. Decision Tree
dt_pred = predict(dt, test[,-16], type = 'class')
dt_table = table(predicted = dt_pred, actual = test$CloudTomorrow)
dt_con_mat = caret::confusionMatrix(dt_table, positive = "1")
dt_con_mat


#2. Logistic Rgression Classifier
# try different thresholds
glm_pred5 = ifelse(predict(model_glm, test[,-16], type = "response") > 0.5, "1", "0")
calc_class_acc(actual = test$CloudTomorrow, predicted = glm_pred5)
glm_pred3 = ifelse(predict(model_glm, test[,-16], type = "response") > 0.3, "1", "0")
calc_class_acc(actual = test$CloudTomorrow, predicted = glm_pred3)
glm_pred1 = ifelse(predict(model_glm, test[,-16], type = "response") > 0.1, "1", "0")
calc_class_acc(actual = test$CloudTomorrow, predicted = glm_pred1)

#Using threshold 0.5 to determine the class of logistic regression classification gets the best result. So we use this value for model comparison.

glm_table = table(predicted = glm_pred5, actual = test$CloudTomorrow)
glm_con_mat = caret::confusionMatrix(glm_table, positive = "1")
glm_con_mat


#3. Random Forest
rf_pred = predict(rf_mod, test[,-16], type = 'class')
rf_table = table(predicted = rf_pred, actual = test$CloudTomorrow)
rf_con_mat = caret::confusionMatrix(rf_table, positive = "1")
rf_con_mat


#4. Bagged Tree
bag_pred = predict(bag, test[,-16], type = 'class')
bag_table = table(predicted = bag_pred, actual = test$CloudTomorrow)
bag_con_mat = caret::confusionMatrix(bag_table, positive = "1")
bag_con_mat


#5. LDA
lda_pred = predict(clf_lda, test[,-16], type = 'class')
lda_table = table(predicted = lda_pred$class, actual = test$CloudTomorrow)
lda_con_mat = caret::confusionMatrix(lda_table, positive = "1")
lda_con_mat

#6. QDA
qda_pred = predict(clf_qda, test[,-16], type = 'class')
qda_table = table(predicted = qda_pred$class, actual = test$CloudTomorrow)
qda_con_mat = caret::confusionMatrix(qda_table, positive = "1")
qda_con_mat


### 3.2
#Compare the classifiers with ROC curves and label the AUCs.

# Plot the ROC curves in one plot
### Get the predictions from each model for plotting the ROC curve
dt_test_prob = predict(dt, newdata = test[,-16], type = "prob")
test_prob = predict(model_glm, newdata = test[,-16], type = "response")
rf_test_prob = predict(rf_mod, newdata = test[,-16], type = "prob")
bag_test_prob = predict(bag, newdata = test[,-16], type = "prob")
lda_test_prob = predict(clf_lda, newdata = test[,-16], type = "prob")
qda_test_prob = predict(clf_qda, newdata = test[,-16], type = "prob")

roc(test$CloudTomorrow, dt_test_prob[,2], plot = TRUE, print.auc = TRUE, legacy.axes=TRUE,percent = TRUE, xlab="False Positive Percentage", ylab="True Positive Percentage", col="#1b9e77", main="ROC Curves",print.auc.y=45, lwd = 2)
lines.roc(test$CloudTomorrow, test_prob, plot = TRUE, print.auc = TRUE,percent = TRUE, add = TRUE, print.auc.y=40,col="#d95f02",lwd = 2)
lines.roc(test$CloudTomorrow, rf_test_prob[,2], plot = TRUE, print.auc = TRUE,percent = TRUE, add = TRUE, print.auc.y=35,col="#7570b3",lwd = 2)
lines.roc(test$CloudTomorrow, bag_test_prob[,2], plot = TRUE, print.auc = TRUE,percent = TRUE, add = TRUE, print.auc.y=30,col="#e7298a",lwd = 2)
lines.roc(test$CloudTomorrow, lda_test_prob$posterior[,2], plot = TRUE, print.auc = TRUE,percent = TRUE, add = TRUE, print.auc.y=25,col="#66a61e",lwd = 2)
lines.roc(test$CloudTomorrow, qda_test_prob$posterior[,2], plot = TRUE, print.auc = TRUE,percent = TRUE, add = TRUE, print.auc.y=20,col="#e6ab02",lwd = 2)
# Add legend
legend("bottomright", legend = c("DT","Logistic","RF","Bag","LDA","QDA"), col=c("#1b9e77","#d95f02","#7570b3","#e7298a","#66a61e","#e6ab02"),lwd = 3)

#Through the ROC curves, Bagged tree has the best True Positive rate, and the second one is random forest. The closer to the top-left corner of curvature, the higher the sensitivity is. We can also get the same result by obersering the values from the comfusion marix. The further comparison of results would be discussed in session 3.5. In classification tasks, though the accuracy is important, we shpuld also consider two important conditions:
#1. Task's characteristics:
#Take spam as an example, classified an important email would make more trouble than misclassify spam to normal email. So true positive rate and accuracy are not always the golden rule.
#2.Imbalances classes problem:
#If the dataset has imbalanced classes, it's easy for classifiers to achieve high accuracy. However, this could caused by only classify the samples which are not belong to the target class, and the classifiers ccould not actually tell the real targets.

### 3.3 10-Fold Cross Validation for each model
#1. Decision Tree
set.seed(123) 
train.control <- trainControl(method = "cv", number = 10, savePredictions=TRUE)
dt_cv <- train(CloudTomorrow ~., data = train, method = "rpart",
               trControl = train.control)
print(dt_cv$resample)

#2. Logistic Rgression Classifier
lgr_cv <- train(CloudTomorrow ~., data = train, method = "glm", trControl = train.control)
print(lgr_cv$resample)


#3. Random Forest
mtry <- sqrt(ncol(train[,-16]))
tunegrid <- expand.grid(.mtry=mtry)
rf_cv <- train(CloudTomorrow ~., data = train, method = "rf",trControl = train.control, tuneGrid=tunegrid)
print(rf_cv$resample)


#4. Bagged Tree
bag_cv <- train(CloudTomorrow ~., data = train, method = "treebag",trControl = train.control)
print(bag_cv$resample)


#5. LDA
lda_cv <- train(CloudTomorrow ~ Sunshine + Humidity3pm + Location + Humidity9am + Pressure9am + Pressure3pm + MinTemp + MaxTemp + Day + Year + Month, data = train, method = "lda", trControl = train.control)
print(lda_cv$resample)


#6. QDA
qda_cv <- train(CloudTomorrow ~ Sunshine + Humidity3pm + Location + Humidity9am + Pressure9am + Pressure3pm + MinTemp + MaxTemp + Day + Year + Month, data = train, method = "qda", trControl = train.control)
print(qda_cv$resample)

#All classifiers get a stable result from it's 10-fold cross validation.


### 3.4 Classifeir Performance Comparision
print("DecisionTree Classifier:")
dt_con_mat$table
print("-------------")
print(paste("Accuracy:", dt_con_mat$overall['Accuracy']))
print("AUC=64.5%")
print("========================")

print("Logistic Classifier:")
glm_con_mat$table
print("-------------")
print(paste("Accuracy:", glm_con_mat$overall['Accuracy']))
print("AUC=69.9%")
print("========================")

print("RandomForest Classifier:")
rf_con_mat$table
print("-------------")
print(paste("Accuracy:", rf_con_mat$overall['Accuracy']))
print("AUC=76.8%")
print("========================")

print("Bag Classifier:")
bag_con_mat$table
print("-------------")
print(paste("Accuracy:", bag_con_mat$overall['Accuracy']))
print("AUC=77.2%")
print("========================")

print("LDA Classifier:")
lda_con_mat$table
print("-------------")
print(paste("Accuracy:", lda_con_mat$overall['Accuracy']))
print("AUC=69.2%")
print("========================")

print("QDA Classifier:")
qda_con_mat$table
print("-------------")
print(paste("Accuracy:", qda_con_mat$overall['Accuracy']))
print("AUC=69.4%")
print("========================")

#From the comparision table, decision tree has the lowest accuracy according to the value of test accuracy and the traing AUC shows the same fact. However, the confusion matric shows that decision tree can best tell the cloudy days though it has the worst ability to tell the clear days. And looking into the result of QDA model, it has the worst performance although it doesn't have the worst accuracy. It can only tell more days which are clear instead of telling more cloudy days. This doesn't meet the need of out main purpose. 
#In our case, bagged tree is the best classifier in the analysis, it not only has the highest accuracy on testing data, but also get the best ROC curve. Besides, the ability to get the true positive cases is slightly worse than decision tree, the result is accaptable.


### 3.5
#According to the result of the decision tree, predictor Sunshine can best seperate the data into two classes, and the second important predictors is Location. This can be verified from the importance plot of random forest. Then, when looking into LDA's coefficient, Sunshine also gets the higest value, but the rest predictors ranking is a bit different from tree methods. 
#In the end, all the predictors about winds are not so informative to the prediction, and these categorical data cause the collinearity problem which would make a serious impact on linear models. And we can also see it through the LDA and QDA modles if we don't discard these variables.



