# -*- coding: utf-8 -*-
"""
Created on Thu May 30 22:20:07 2019

@author: NR
"""
#importing the library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset using pandas
dataset = pd.read_csv('housing.csv')

#separating the features and label
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,9].values

#dealing with missing value
from sklearn.preprocessing import Imputer
missingValueImputer = Imputer (missing_values = 'NaN', strategy = 'mean', 
                               axis = 0)
missingValueImputer = missingValueImputer.fit (X[:,4:5])
X[:,4:5] = missingValueImputer.transform(X[:,4:5])

#encoding the dataset
from sklearn.preprocessing import LabelEncoder
X_labelencoder = LabelEncoder()
X[:, 8] = X_labelencoder.fit_transform(X[:, 8])
print (X)

# Implementing OneHotEncoder to separate category variables into dummy 
# variables.
#==============================================================================
from sklearn.preprocessing import OneHotEncoder
X_onehotencoder = OneHotEncoder (categorical_features = [8])
X = X_onehotencoder.fit_transform(X).toarray()

print (X)

#spilt the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.2, random_state = 0)


#feature scaling 

from sklearn.preprocessing import StandardScaler
independent_scalar = StandardScaler()
X_train = independent_scalar.fit_transform (X_train) 
X_test = independent_scalar.transform (X_test)

#fitiing the linear regression

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predecting the value
y_pred = regressor.predict(X_test)

regressor.score(X_train,y_train)
regressor.score(X_test,y_test)

#RMSE
from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(y_test,y_pred)
print(rmse)
rmse1 = mean_squared_error(y_train,regressor.predict(X_train))
print(rmse1)

# *************************************************
#   fitting the dicision tree 
# *********************************

from sklearn.tree import DecisionTreeRegressor
DTRegressor = DecisionTreeRegressor(random_state=0)
DTRegressor.fit(X_train,y_train)

#predicting the value
y_predDT = DTRegressor.predict(X_test)

#checking the socre 
DTRegressor.score(X_train,y_train)
DTRegressor.score(X_test,y_test)

#RMSE
from sklearn.metrics import mean_squared_error
cm1 = mean_squared_error(y_test,y_predDT)
print(cm1)
cm2 = mean_squared_error(y_train,DTRegressor.predict(X_train))
print(cm2)

#fitting the randomforest

from sklearn.ensemble import RandomForestRegressor
regressor1 = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor1.fit(X_train, y_train)

#predectiion 
y_predRF = regressor1.predict(X_test)

regressor1.score(X_train,y_train)
regressor1.score(X_test,y_test)

#RMSE
from sklearn.metrics import mean_squared_error
rmseR = mean_squared_error(y_test,y_predRF)
print(rmseR)
rmseR1 = mean_squared_error(y_train,regressor1.predict(X_train))
print(rmseR1)

# **************************************************************
#Perform Linear Regression with one independent variable :
# *************************************************************

X_train1 = X_train[:,[12]]
X_test1 =  X_test[:,[12]]


#fitiing the linear regression

from sklearn.linear_model import LinearRegression
regressor2 = LinearRegression()
regressor2.fit(X_train1,y_train)

#predecting the value
y_pred = regressor2.predict(X_test1)

#checking the score
regressor2.score(X_train1,y_train)
regressor2.score(X_test1,y_test)

#visuliztion trining data 

plt.scatter(X_train1,y_train,color='red')
plt.plot(X_train1,regressor2.predict(X_train1),color='blue')
plt.xlabel('Medain Income')
plt.ylabel('House Price')
plt.title('Visualization for training dataset')
plt.show()

#for test dataset
plt.scatter(X_test1,y_test,color='red')
plt.plot(X_train1,regressor2.predict(X_train1),color='blue')
plt.xlabel('Medain Income')
plt.ylabel('House Price')
plt.title('Visualization for training dataset')
plt.show()
