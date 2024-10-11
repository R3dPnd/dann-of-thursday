# Data Preprocessing

There is typically a step by step process for analysis of data.

* Data Preprocessing
  * Importing data
  * clean the data
  * split into training sets
* Modelling
  * Build the model
  * Train the model
  * Make Prediction
* Evaluation
  * Calculate performance metrics
  * Make a verdict

## Training Set and Test Set

Whenever analyzing data you will have dependent variables and independent variables. Splitting data implys moving some data out before processing as the data is not relevant or it will negatively impact our analysis.

Train data is the data being passed in that we are trying to learn from, where as the Test data is finalized data we can compare the results from the training to.

## Feature Scaling

Feature scaling is applied across columns in a data set. there are two main techincews

* Normalization - taking the min of a column and subtracting this from each value in the column and then dividing this value by the difference between min and max values
* Standardization
