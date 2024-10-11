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
dataset = pd.read_csv('02-Regression\Salary_Data.csv')

# iloc locates indexes of rows and columns, we are making a subset of the dataset
# to get all the rows without knowing the full range, we use : for the rows
# To get all but the last one, we use -1 as the end index
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# This will give us the sets needed for training and testing the model
# We are giving an 20 test 80 train split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)