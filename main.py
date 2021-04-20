import pandas as pd
import numpy as np
! pip install ipywidgets
from ipywidgets import interact, interactive, fixed, interact_manual

# Import clean data 
path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/module_5_auto.csv'
df = pd.read_csv(path)

df.to_csv('module_5_auto.csv')

# Let's get an understanding of the data
df=df._get_numeric_data()
df.head()

# Get dependent (y_data) and independent (x_data) variables
y_data = df['price']
x_data=df.drop('price',axis=1)

# Randomly split our data into training and testing data
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10, random_state=1)

print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])

# Lets create a Linear Regression and see the difference between the R^2 of the test and train data
from sklearn.linear_model import LinearRegression
lre=LinearRegression()
lre.fit(x_train[['horsepower']], y_train)
lre.score(x_test[['horsepower']], y_test)
lre.score(x_train[['horsepower']], y_train)
# -> R^2 is much smaller on the test data

# Lets get the Cross Validation Score
from sklearn.model_selection import cross_val_score
Rcross = cross_val_score(lre, x_data[['horsepower']], y_data, cv=4)
Rcross
print("The mean of the folds are", Rcross.mean(), "and the standard deviation is" , Rcross.std())
