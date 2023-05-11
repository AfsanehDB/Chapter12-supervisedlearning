# Chapter12-supervisedlearning
# Module 12 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
* Explain what financial information the data was on, and what you needed to predict.
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
* Describe the stages of the machine learning process you went through as part of this analysis.
* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).
    Credit risk poses a classification problem that’s inherently imbalanced. This is because healthy loans easily outnumber risky loans. In this Challenge, we’ll use various techniques to train and evaluate models with imbalanced classes. we’ll use a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.
data was used includ information on "loan_size, interest_rate, borrower_income, debt_to_income, num_of_accounts, derogatory_marks, total_debt and loan_status. 
These data will be used to build  a modle thta can pridict which loan will defult. we will use "Logistic Regression and Over sample" methodes to build the model. Here are the steps we will take:

This challenge consists of the following subsections:

* Split the Data into Training and Testing Sets

* Create a Logistic Regression Model with the Original Data

* Predict a Logistic Regression Model with Resampled Training Data 

### Split the Data into Training and Testing Sets

Open the starter code notebook and then use it to complete the following steps.

1. Read the `lending_data.csv` data from the `Resources` folder into a Pandas DataFrame.

2. Create the labels set (`y`)  from the “loan_status” column, and then create the features (`X`) DataFrame from the remaining columns.

    > **Note** A value of `0` in the “loan_status” column means that the loan is healthy. A value of `1` means that the loan has a high risk of defaulting.  

3. Check the balance of the labels variable (`y`) by using the `value_counts` function.

4. Split the data into training and testing datasets by using `train_test_split`.

### Create a Logistic Regression Model with the Original Data

1. Fit a logistic regression model by using the training data (`X_train` and `y_train`).

2. Save the predictions on the testing data labels by using the testing feature data (`X_test`) and the fitted model.

3. Evaluate the model’s performance by doing the following:

    * Calculate the accuracy score of the model.

    * Generate a confusion matrix.

    * Print the classification report.

4. Answer the following question: How well does the logistic regression model predict both the `0` (healthy loan) and `1` (high-risk loan) labels?

### Predict a Logistic Regression Model with Resampled Training Data

Did you notice the small number of high-risk loan labels? Perhaps, a model that uses resampled data will perform better. You’ll thus resample the training data and then reevaluate the model. Specifically, you’ll use `RandomOverSampler`.

To do so, complete the following steps:

1. Use the `RandomOverSampler` module from the imbalanced-learn library to resample the data. Be sure to confirm that the labels have an equal number of data points. 

2. Use the `LogisticRegression` classifier and the resampled data to fit the model and make predictions.

3. Evaluate the model’s performance by doing the following:

    * Calculate the accuracy score of the model.

    * Generate a confusion matrix.

    * Print the classification report.
    
4. Answer the following question: How well does the logistic regression model, fit with oversampled data, predict both the `0` (healthy loan) and `1` (high-risk loan) labels?
## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.



* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )
result from the 2 methodes are as follow:
The LR it predict 56000  case of '0'(Good Loan) and 1676 case of '1' (bad Loan) correctly. with balanced_accuracy_score 94%. for good loans it predict 100% cases accurate and bad loan 87%.  

Logistic Regrassion: 
0.9434928322178853
[56000   289]
[  187  1676]
testing_predictions
print(classification_report_imbalanced(y_test, y_predictions))
                   pre       rec       spe        f1       geo       iba       sup

          0       1.00      1.00      0.89      1.00      0.94      0.90     18747
          1       0.87      0.89      1.00      0.88      0.94      0.88       637

avg / total       0.99      0.99      0.90      0.99      0.94      0.90     19384


Oversampling:
0.9950047894633314

[18648    99]
[    3   634]

                   pre       rec       spe        f1       geo       iba       sup

          0       1.00      0.99      1.00      1.00      1.00      0.99     18747
          1       0.86      1.00      0.99      0.93      1.00      0.99       637

avg / total       1.00      0.99      1.00      0.99      1.00      0.99     19384

If you do not recommend any of the models, please justify your reasoning.
Notice that the balanced accuracy score is 94.3%. Comparing that to the balanced accuracy score of 99.5 for the oversampled data, we can observe that the balanced accuracy score for the oversampled data is higher. But, not that much biger.
The model that used the imbalanced data produced a precision of 0.87 The model that used the oversampled data produced a precision of 0.86. So, there is no significant diference at making predictions for the 1 class.
Finally, let’s compare how the two models did with the recall for the 1 class. The model that used the imbalanced data produced a recall of 0.89 The model that used the oversampled data produced a recall of 1. So, the model that used the oversampled data was sligtly more accurate at predicting who would both download and use the application.


The model that used the imbalanced data produced a precision of 1.The model that used the oversampled data produced a precision of 1. So, there is no diference at making predictions for the 1 class.
Finally, let’s compare how the two models did with the recall for the 1 class. The model that used the imbalanced data produced a recall of 1. The model that used the oversampled data produced a recall of .99 So, the model that RL data was sligtly more accurate at predicting who would both download and use the application.
overall both methodes predict the result very well and there is no big diferrence between the pridiction. 
