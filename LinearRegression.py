import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Import CSV file
TD = pd.read_csv('TrainData.csv')

#
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

max_x = np.max(X)
min_x = np.min(X)

x = np.linspace(min_x, max_x, 10)
y = b0 + b1*x

print(b0, b1)

plt.plot(x,y, color='r', label='regression line')

plt.scatter(X,Y,color='b', label='Scatter')

plt.xlabel('Wind Speed')
plt.ylabel('Power')
plt.ylim(0,1)
plt.legend()
plt.show()

X = X.reshape((n,1))

reg = LinearRegression()

reg = reg.fit(X,Y)

Y_pred = reg.predict(X)

rmse = reg.score(X,Y)

print('rmse score is', rmse)

#Forecast of future generation
WFI = pd.read_csv('WeatherForecastInput.csv')
SOL = pd.read_csv('Solution.csv')

WFI['POWER'] = (WFI['WS10']*b1+b0)
WFI['ActPower'] = SOL['POWER']


x = WFI['TIMESTAMP'].values
y = WFI['POWER']
y2 = WFI['ActPower']

plt.plot(x,y, color = 'r', label ='Predicted Power')
plt.plot(x,y2, color = 'g', label ='Actual Power')
plt.legend()
plt.rcParams['figure.figsize'] = [16, 12]
plt.ylabel('Power')
plt.xlabel('Time')
plt.show()