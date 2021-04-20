# Cross-Validation-Score-Linear-Regression-Python

The in-sample evaluation tells us how well our model will fit the data used to train it.
However, it does not tell us how well the trained model can be used to predict new data.
Backtesting is necessary.
Therfore, the known data must be split into training and testing data.
In order to get a representative result we must use cross validation.
In cross validation a loop is executed in which various training data and testing data are used and checked.
The average result of this loop indicates how well the model is doing for the forecast.
Check out the Python source code for the implementation.
Nevertheless, black swan events, insufficient test data, etc. can make the result invalid, even with a good cross validation score.
Creating the future is always the best way to predict it.
