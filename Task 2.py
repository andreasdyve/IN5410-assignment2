import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
import numpy as np
from sklearn.metrics import mean_squared_error
df_train = pd.read_csv('TrainData.csv')

# Calculating and creating columns for wind direction

df_train['$\phi$'] = np.arctan(df_train['V10']/df_train['U10'])

def f(row):
    if row['U10'] <= 0 and row['V10'] <= 0:
        val = 2*np.pi - row['$\phi$']
        
    elif row['U10'] > 0 and row['V10'] <= 0:
        val = row['$\phi$']
        
    elif row['U10'] > 0 and row['V10'] > 0:
        val = np.pi - row['$\phi$']
        
    elif row['U10'] <= 0 and row['V10'] > 0:
        val = np.pi + row['$\phi$']
    
    else: val = 99
        
    return val

df_train['$\theta$'] = df_train.apply(f, axis=1)

df_train['hour'] = np.arange(1, 16081, 1)

# Training the model on the training set
x_train = df_train[['WS10', '$\theta$']]
y_train = df_train['POWER']

model = linear_model.LinearRegression()
model.fit(x_train, y_train)

# Importing Test data
WFI = pd.read_csv('WeatherForecastInput.csv')
SOL = pd.read_csv('Solution.csv')

#Creating dunf direction columns for test data
WFI['$\phi$'] = np.arctan(WFI['V10']/WFI['U10'])

def f(row):
    if row['U10'] <= 0 and row['V10'] <= 0:
        val = 2*np.pi - row['$\phi$']
        
    elif row['U10'] > 0 and row['V10'] <= 0:
        val = row['$\phi$']
        
    elif row['U10'] > 0 and row['V10'] > 0:
        val = np.pi - row['$\phi$']
        
    elif row['U10'] <= 0 and row['V10'] > 0:
        val = np.pi + row['$\phi$']
    
    else: val = 99
        
    return val

WFI['$\theta$'] = WFI.apply(f, axis=1)

WFI['hour'] = np.arange(1, 721, 1)

#POWER PREDICTION
x_test = WFI[['WS10', '$\theta$']]
y_test = SOL['POWER']
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(mse)
print(rmse)

#Plotting
WFI['Predicted Power'] = y_pred

plt.plot(WFI['hour'], y_test, 'y', label='Actual Power (Solution)')
plt.plot(WFI['hour'], y_pred, 'blue', label='Predicted Power (MLR)')
plt.rcParams['figure.figsize'] = [16, 12]
plt.ylabel('Power')
plt.xlabel('Time[hours]')
plt.title('MLR November 2013')
plt.legend()
plt.savefig('MLR_whole_nov')
plt.show()

#MLR
df2_train = df_train.copy()
x2_train = df2_train['WS10'].values.reshape(-1,1)
y2_train = df2_train['POWER']

model2 = linear_model.LinearRegression()
model2.fit(x2_train, y2_train)

x2_test = WFI['WS10'].values.reshape(-1,1)
y2_test = SOL['POWER']
y2_pred = model2.predict(x2_test)
mse2 = mean_squared_error(y2_test, y2_pred)
rmse2 = np.sqrt(mse2)
print(mse2)
print(rmse2)

plt.plot(WFI['hour'], y2_test, 'y', label='Actual Power (Solution)')
plt.plot(WFI['hour'], y_pred, 'blue', label='Predicted Power (MLR)')
plt.plot(WFI['hour'], y2_pred, 'red', label='Predicted Power (LR)')
plt.rcParams['figure.figsize'] = [16, 12]
plt.ylabel('Power')
plt.xlabel('Time[hours]')
plt.title('MLR vs LR November 2013')
plt.legend()
plt.savefig('MLR_vs_LR_whole_nov')
plt.show()