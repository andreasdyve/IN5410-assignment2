import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics, svm
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import math
from sklearn import datasets, linear_model

TD = pd.read_csv('TrainData.csv')
TD['Index'] = TD.index

# Linear Regression
y = TD['POWER']
x = TD['WS10']

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

#print(b1, b0)

max_x = np.max(X)
min_x = np.min(X)

x = np.linspace(min_x, max_x, 10)
y = b0 + b1*x

plt.plot(x,y, color='r', label='regression line')

plt.scatter(X,Y,s=0.5,color='b', label='Scatter')

plt.xlabel('Wind Speed (m/s)')
plt.ylabel('Power')
plt.ylim(0,1)
plt.legend()
plt.title('LR predicted power')
plt.show()

WFI = pd.read_csv('WeatherForecastInput.csv')
SOL = pd.read_csv('Solution.csv')

WFI['POWER'] = (WFI['WS10']*b1+b0)
WFI['ActPower'] = SOL['POWER']
WFI['Index'] = WFI.index

x = WFI['Index'].values
y_pred = WFI['POWER']
y_test = WFI['ActPower']

plt.plot(x,y_pred, color = 'b', label ='Predicted Power (LR)')
plt.plot(x,y_test, color = 'y', label ='Actual Power')
plt.legend()
plt.rcParams['figure.figsize'] = [16, 12]
plt.ylabel('Power')
plt.xlabel('Time [Hours]')
plt.title('LR Predicted Power November 2013')
plt.show()

np.savetxt("ForecastTemplate1-LR.csv", y, delimiter=",")

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:",mse)
rmse = math.sqrt(mse)
print("Root Mean Squared Error LR:", rmse)

#Plots for weekly predictions excluded in this script

# kNN method
KNN = TD.copy()
WFI_1 = pd.read_csv('WeatherForecastInput.csv')
SOL_1 = pd.read_csv('Solution.csv')
WFI_1['Index'] = WFI_1.index

WFI_1['POWER'] = SOL_1['POWER']

KNN = KNN[['TIMESTAMP', 'POWER', 'U10', 'V10', 'WS10']]
WFI_1 = WFI_1[['TIMESTAMP', 'POWER', 'U10', 'V10', 'WS10','Index']]

# KNN is train data
# WFI_1 is test data

X_train = KNN.iloc[:, 4].values #WS10
X_train = X_train.reshape(-1,1)
y_train = KNN.iloc[:, 1].values #y_train is POWER
y_train = y_train.reshape(-1,1)

X_test = WFI_1.iloc[:, 4].values #WS10
X_test = X_test.reshape(-1,1)
y_test = WFI_1.iloc[:, 1].values #y_test is POWER

clf = KNeighborsRegressor(n_neighbors=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

x = WFI_1['WS10']
y = WFI_1['POWER']
plt.scatter(x,y_pred, label = 'Predicted Power (kNN)')
plt.scatter(x, y, label = 'Actual Power', color = 'y')
plt.legend()
plt.rcParams['figure.figsize'] = [16, 12]
plt.ylabel('Power')
plt.xlabel('Wind Speed (m/s)')
plt.title('kNN Predicted vs. Actual Power')
plt.show()

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:",mse)
rmse = math.sqrt(mse)
print("Root Mean Squared Error kNN:", rmse)

np.savetxt("ForecastTemplate1-kNN.csv", y_pred, delimiter=",")

x1 = WFI_1.iloc[0:720]['Index']
y = WFI_1.iloc[0:720]['POWER']
y_2 = y_pred[0:720]
plt.plot(x1, y, label = 'Actual Power (Solution)', color = 'y')
plt.plot(x1, y_2, label = 'Predicted Power (kNN)')
plt.ylabel('Power')
plt.xlabel('Time')
plt.legend()
plt.title('kNN Predicted Power November 2013')
plt.show()

# SVR method

SVR = TD.copy()
WFI_2 = pd.read_csv('WeatherForecastInput.csv')
SOL_2 = pd.read_csv('Solution.csv')
WFI_2['Index'] = WFI_2.index

WFI_2['POWER'] = SOL_2['POWER']

SVR = SVR[['TIMESTAMP', 'POWER', 'U10', 'V10', 'WS10']]
WFI_2 = WFI_2[['TIMESTAMP', 'POWER', 'U10', 'V10', 'WS10','Index']]

# SVR is train data
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
plt.rcParams['figure.figsize'] = [16, 12]
plt.ylabel('Power')
plt.xlabel('Wind Speed (m/s)')
plt.title('SVR Predicted Power')
plt.show()

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:",mse)
rmse = math.sqrt(mse)
print("Root Mean Squared Error SCR:", rmse)

np.savetxt("ForecastTemplate1-SVR.csv", y_pred, delimiter=",")

x1 = WFI_2['Index']
y = WFI_2['POWER']
y_3 = y_pred
plt.plot(x1, y, label = 'Actual Power', color = 'y')
plt.plot(x1, y_3, label = 'Predicted Power (SVR)')
plt.ylabel('Power')
plt.xlabel('Time [Hours]')
plt.title('SVR Predicted Power November 2013')
plt.legend()
plt.show()

#ANN method

ANN = TD.copy()

WFI_3 = pd.read_csv('WeatherForecastInput.csv')
SOL_3 = pd.read_csv('Solution.csv')
WFI_3['Index'] = WFI_3.index

WFI_3['POWER'] = SOL_3['POWER']

ANN = ANN[['TIMESTAMP', 'POWER', 'U10', 'V10', 'WS10']]
WFI_3 = WFI_3[['TIMESTAMP', 'POWER', 'U10', 'V10', 'WS10','Index']]

# KNN is train data
# WFI_1 is test data

X_train = ANN.iloc[:, 4].values #X_Train in WS10
X_train = X_train.reshape(-1, 1)
y_train = ANN.iloc[:, 1].values #y_train is POWER
y_train = y_train.reshape(-1, 1)

X_test = WFI_3.iloc[:, 4].values #X_test is WS10
X_test = X_test.reshape(-1, 1)

y_test = WFI_3.iloc[:, 1].values #y_test is POWER from solution
y_test = y_test.reshape(-1, 1)

mlp = MLPRegressor(max_iter = 1000, activation = 'relu')
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)

x = WFI_3['WS10']
plt.scatter(x,y_pred, label = 'Predicted Power (ANN)')
plt.scatter(x, y_test, label = 'Actual Power', color = 'y')
plt.legend()
plt.rcParams['figure.figsize'] = [16, 12]
plt.ylabel('Power')
plt.xlabel('Wind Speed (m/s)')
plt.title('ANN Predicted Power')
plt.show()

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:",mse)
rmse = math.sqrt(mse)
print("Root Mean Squared Error ANN:", rmse)

np.savetxt("ForecastTemplate1-ANN.csv", y_pred, delimiter=",")

x1 = WFI_3['Index']
y = WFI_3['POWER']
y_4 = y_pred
plt.plot(x1, y, label = 'Actual Power', color = 'y')
plt.plot(x1, y_4, label = 'Predicted Power (ANN)')
plt.ylabel('Power')
plt.xlabel('Time [Hours]')
plt.title('ANN Predicted Power November 2013')
plt.legend()
plt.show()