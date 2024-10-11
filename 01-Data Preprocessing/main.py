import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importing the dataset
dataset = pd.read_csv('01-Data Preprocessing\Data.csv')

# The independent variables are the Country, Age, and Salary
# The dependent variable is the Purchased, based on the other 3 variables

# iloc locates indexes of rows and columns, we are making a subset of the dataset
# to get all the rows without knowing the full range, we use : for the rows
# To get all but the last one, we use -1 as the end index
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Taking care of missing data using the average
# Replaces the given value with the strategy provided, in this case the mean
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
# print(X)

# Encoding Categorical Data
# There are 3 countries and we want to make 3 catagories based on this
# These will bee converted into vectors

# Supply the transformers and the columns to transform
# The transformer is supplied with type of transformation, the encoder to use and the columns to transform
# The remainder is set to passthrough to keep the columns not transformed
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# returns a new matrix with the transformed columns
X = np.array(ct.fit_transform(X))
# print(X)

# Encoding the Dependent Variable
# Assigning numeric values to the dependent variable, 0 for no and 1 for yes
le = LabelEncoder();
Y = le.fit_transform(Y)
# print(Y)

# Feature Scaling
# Feature Scaling should happen after splitting the data into training and test sets
# Splitting the data mean creating one set for training the data and one for evaluating the trained model
# We need to scale the data to make sure that the model is not biased towards one feature
# The reason we need to scale after is to prevent information leakage into the test set

# This will give us the sets needed for training and testing the model
# We are giving an 20 test 80 train split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# Feature Scaling can be done through standardization, 
# taking the value minus the average divided by the standard deviation, 
# or normalization, value minus the minimum divided by the range
# Standardization is generally very good, while Normalization is only really good for data following a normal distribution
sc = StandardScaler()
# Taking all the rows and only columns 1 and 3
# This will compute the mean and std dev for the selected columns
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_train[:, 3:])
print(X_train)
print(X_test)

# We can use this data to create regression sets