#IN5410 - Assignment 2 - Task 1
# Group 1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model

TD = pd.read_csv('TrainData.csv')

##########################################
#Linear Regression
##########################################

X = TD['WS10'].values
Y = TD['POWER'].values

mean_X = np.mean(X)
mean_Y = np.mean(Y)

n = len(X)

numer = 0
denom = 0
for i in range(n):
    numer += (X[i] - mean_X) * (Y[i] - mean_Y)
    denom += (X[i] - mean_X) ** 2
b1 = numer / denom
b0 = mean_Y - (b1 * mean_X)

print(b1, b0)

max_x = np.max(X)
min_x = np.min(X)

x = np.linspace(min_x, max_x, 10)
y = b0 + b1*x

plt.plot(x,y, color='r', label='regression line')
plt.scatter(X,Y,color='b', label='Scatter')
plt.title('Linear Regression Line')
plt.xlabel('Wind Speed')
plt.ylabel('Power')
plt.ylim(0,1)
plt.legend()
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = X.reshape((n,1))

reg = LinearRegression()

reg = reg.fit(X,Y)

Y_pred = reg.predict(X)

rmse = reg.score(X,Y)

print('Linear Regression RMSE is', rmse)

WFI = pd.read_csv('WeatherForecastInput.csv')
SOL = pd.read_csv('Solution.csv')

WFI['POWER'] = (WFI['WS10']*b1+b0)
WFI['ActPower'] = SOL['POWER']

x = WFI['TIMESTAMP'].values
y = WFI['POWER']
y2 = WFI['ActPower']

plt.plot(x,y, color = 'r', label ='Predicted Power (LR)')
plt.plot(x,y2, color = 'g', label ='Actual Power')
plt.legend()
plt.ylabel('Power')
plt.xlabel('Time')
plt.title('LR Predicted Power and Actual Power November 2013')
plt.show()

# np.savetxt("ForecastTemplate1-LR.csv", y, delimiter=",")

##########################################
#k Nearest Neighbor
##########################################

KNN = TD.copy()
WFI_1 = pd.read_csv('WeatherForecastInput.csv')
SOL_1 = pd.read_csv('Solution.csv')

WFI_1['POWER'] = SOL_1['POWER']

KNN = KNN[['TIMESTAMP', 'POWER', 'U10', 'V10', 'WS10']]
WFI_1 = WFI_1[['TIMESTAMP', 'POWER', 'U10', 'V10', 'WS10']]

# KNN is train data
# WFI_1 is test data

X_train = KNN.iloc[:, 4].values #X_train is WS10
X_train = X_train.reshape(-1,1)
y_train = KNN.iloc[:, 1].values #y_train is POWER
y_train = y_train.reshape(-1,1)

X_test = WFI_1.iloc[:, 4].values #X_test is WS10
X_test = X_test.reshape(-1,1)
y_test = WFI_1.iloc[:, 1].values #y_test is POWER

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
clf = KNeighborsRegressor(n_neighbors=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

x = WFI_1['WS10']
y = WFI_1['POWER']
plt.scatter(x,y_pred, label = 'Predicted Power')
plt.scatter(x, y, label = 'Actual Power', color = 'y')
plt.title('kNN predicted power')
plt.legend()
plt.ylabel('Power')
plt.xlabel('Wind Speed')
plt.show()

from sklearn.metrics import mean_squared_error
import math

mse = mean_squared_error(y, y_pred)
print("Mean Squared Error:",mse)
rmse = math.sqrt(mse)
print("Root Mean Squared Error kNN:", rmse)

# np.savetxt("ForecastTemplate1-kNN.csv", y_pred, delimiter=",")

x1 = WFI_1.iloc[0:720]['TIMESTAMP']
y = WFI_1.iloc[0:720]['POWER']
y_2 = y_pred[0:720]
plt.plot(x1, y, label = 'Actual Power (Solution)', color = 'y')
plt.plot(x1, y_2, label = 'Predicted Power (kNN)')
plt.title('kNN Predicted Power and Actual Power November 2013')
plt.ylabel('Power')
plt.xlabel('Time')
plt.legend()
plt.show()

##########################################
# SVR
##########################################
from sklearn import metrics, svm

SVR = TD.copy()
WFI_2 = pd.read_csv('WeatherForecastInput.csv')
SOL_2 = pd.read_csv('Solution.csv')

WFI_2['POWER'] = SOL_2['POWER']

SVR = SVR[['TIMESTAMP', 'POWER', 'U10', 'V10', 'WS10']]
WFI_2 = WFI_2[['TIMESTAMP', 'POWER', 'U10', 'V10', 'WS10']]

# KNN is train data
# WFI_1 is test data

X_train = SVR.iloc[:, 4].values #X_Train in WS10
X_train = X_train.reshape(-1, 1)
y_train = SVR.iloc[:, 1].values #y_train is POWER
y_train = y_train.reshape(-1, 1)

X_test = WFI_2.iloc[:, 4].values #X_test is WS10
X_test = X_test.reshape(-1, 1)

y_test = WFI_2.iloc[:, 1].values #y_test is POWER from solution
y_test = y_test.reshape(-1, 1)

clf = svm.SVR()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

x = WFI_2['WS10']
plt.scatter(x,y_pred, label = 'Predicted Power (SVR)')
plt.scatter(x, y_test, label = 'Actual Power', color = 'y')
plt.legend()
plt.title('SVR Predicted Power')
plt.ylabel('Power')
plt.xlabel('Wind Speed')
plt.show()

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:",mse)
rmse = math.sqrt(mse)
print("Root Mean Squared Error SVR:", rmse)

#np.savetxt("ForecastTemplate1-SVR.csv", y_pred, delimiter=",")

x1 = WFI_2['TIMESTAMP']
y = WFI_2['POWER']
y_3 = y_pred
plt.plot(x1, y, label = 'Actual Power', color = 'y')
plt.plot(x1, y_3, label = 'Predicted Power (SVR)')
plt.title('SVR Predicted Power and Actual Power November 2013')
plt.ylabel('Power')
plt.xlabel('Time')
plt.legend()
plt.show()