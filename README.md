# assignment

The given dataset has 57 features with no description provided . All the features are numerical and since features are ambiguos , we cannot really draw BUSSINESS insights from it.

Nonetheless I will try to check how data behaves and try to determine most important features for the model. So in accordance with this ,I will use 3 approaches in this notebook :

Logistic Regression and Random Forest
Reduce features with Backward Elimination and train both the model
Train with Full Dataset


First I cleaned the data by dropping duplicates , normalising the data , looking for missing values etc
After that , I used Logistic regression because :

1.It is simple to comprehend
2.It trains quickly and because we have lots of features , we have to take production time cost into the account , we cannot just build complex ANNs just because it is more accurate.
3.We can adjust class weights as a parameter without having to use smoting or oversampling seperately for class imbalance.
4.We get prediction probabilites unlike some other algorithms , because of which we can set probability threshold to classify.
5.It doesnt overfits easily as it's a linear model.

It performed fairly well , but in order to reduce False negatives , I plotted ROC curve and selected a threshold .
Next I used Backward Feature Elimination to remove some features and Model performed even better after that. 

I used same methods for Random forest classifiers

For Class Imbalance , I used Class Weights Params for Both the models.

Conclusions:

1. Many of the Features had many zero values
2. There were 300 duplicates 
3. We Dropped data with high multicollinearity 
4. Logistic regression with adjusted threshold and feature elimination actually improved our accuracy
5. Model didnt overfit , so no need to do Cross Validation and Regularization
6. Logistic regression took way less time than Random Forest
7. On the other hand Random forest gave us better accuracy
8. At the end of the day , we need to experiment more to get even more accuracy
