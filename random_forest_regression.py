# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Load the library with the iris dataset
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss
from sklearn.ensemble import AdaBoostClassifier

# Create an object called iris with the iris data

dataset = pd.read_csv("C:\Users\chira\OneDrive\Desktop\useful-codes\Random_Forest_Regression\Position_Salaries.csv")
# Create a dataframe with the four feature variables

# Importing the dataset
#dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,5]
print(X)
# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Random Forest Regression to the dataset
print(X)
regressor = AdaBoostClassifier(n_estimators = 10, random_state = 0)
regressor.fit(X, y)
feature_importance=pd.Series(regressor.conf_,index=X.columns)
#feature_importance.sort
print(feature_importance)

# Predicting a new result
#y_pred = regressor.predict(6.5)
#print(regressor.feature_importance_)


# Visualising the Random Forest Regression results (higher resolution)
#X_grid = np.arange(min(X), max(X), 0.01)
#X_grid = X_grid.reshape((len(X_grid), 1))
#plt.scatter(X, y, color = 'red')
#plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
#plt.title('Truth or Bluff (Random Forest Regression)')
#plt.xlabel('Position level')
#plt.ylabel('Salary')
#plt.show()