#This program uses an artificial recurrent neural network called Long Short Term memory (LSTM)
# to predict the closing stock price of a corporation (Apple Inc.) using the past 60 day stock price.

#import the libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler #install scikit-learn to import sklearn
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
plt.style.use('fivethirtyeight')

start = datetime(2021, 1, 1)
end = datetime(2022, 11, 21)
#Get the stock quote
stock = '900310'
# stock = '005930.ks'
df = web.DataReader(stock, data_source = 'naver', start = start, end = end)
df = df.astype('float64')
df.isnull().values.any()

#Get the number of rows and columns in the data set
df.shape

#Create a new dataframe with only the 'Close' column
data = df.filter(['Close'])
#Convert the dataframe to a numpy array
dataset = data.values
#Get the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * 0.8)

#Scale the data
scaler = MinMaxScaler(feature_range = (0, 1))
scaled_data = scaler.fit_transform(dataset)

#Create the training data set
#Create the scaled training data set
train_data = scaled_data[0 : training_data_len, :]

#Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60 : i, 0]) # the last day data of indexed i is not included    
    y_train.append(train_data[i, 0]) # the last day data of index i is included

#Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# x_train.shape[0]: number of samples(rows)
# x_train.shape[1]: number of time stamps(columns)
# 1: number of features. ANN takes three dimentional values as inputs


#Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences = True, input_shape = (x_train.shape[1], 1))) #adding a LSTM layer (the first layer) to the model
model.add(LSTM(50, return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))

#Compile the model
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

#Train the model
model.fit(x_train, y_train, batch_size = 1, epochs = 8)

#Create the testing data set
#Create a new array containing scaled values from index 1543 to 2003
test_data = scaled_data[training_data_len - 60 : , :]
#Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len : , :] # the last day data of indexed i is included  
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0]) # the last day data of indexed i is not included  

#Convert the data to a numpy array
x_test = np.array(x_test)

#Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#Get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions) #Inverse transform data

#Get the root mean squared error(RMSE)
rmse = np.sqrt(np.mean(predictions - y_test) ** 2)

#Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

#Calculate tomorrow prices including the past five days
x_recent = test_data[-65:, :]
x_calc = []
for i in range(len(x_recent)-60):
    x_calc.append(x_recent[i:i+60, 0])
x_calc.append(x_recent[-60:, 0])
x_calc = np.array(x_calc)
x_calc = np.reshape(x_calc, (x_calc.shape[0], x_calc.shape[1], 1))
pred_next = model.predict(x_calc)
pred_next = scaler.inverse_transform(pred_next)
pred_next = np.reshape(pred_next, (6))
# day_to_begin = datetime.today() - timedelta(days=4)
# pred_index = [pd.date_range(day_to_begin, periods=6).strftime('%Y-%m-%d')]
pred_index = list(valid.index[-5:])
adding = datetime.today()+timedelta(days=1)
pred_index.append(datetime.timestamp(adding))
pred = pd.DataFrame({'Predictions' : pred_next}, index=valid.index[-5:])
print(pred)
print(pred_next)
# #Calculate the same values as above but using the prior lines of statements (x_test)
# x_test = []
# # the last day data of indexed i is included  
# for i in range(60, len(test_data)):
#     x_test.append(test_data[i-60:i, 0]) # the last day data of indexed i is not included  
# # The following single line statement includes the last day data of test_data, 
# # which will provide independence variables 
# # to predict the last dependent variable for tomorrow
# x_test.append(test_data[-60: , 0]) 
# #Convert the data to a numpy array
# x_test = np.array(x_test)
# #Reshape the data
# x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# #Get the models predicted price values
# predictions = model.predict(x_test)
# predictions = scaler.inverse_transform(predictions) #Inverse transform data


#Visualize the data
plt.figure(figsize = (16, 8))
plt.title('Model')
plt.xlabel('Data', fontsize = 18)
plt.ylabel('Close Price USD ($)', fontsize = 18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc = 'upper right')
plt.show()

#Calculate a predicted price for the next day
print(f"Price Predicted on {data.index[-1]}: ", predictions[-1])
# print(f"Price Predicted on {data.index[-1]}: ", scaled_data[-1] * predictions[-1] / predictions_scaled[-1])
#print(f"Price Predicted on {data.index[-1]}: ", dataset[-1] / x_test[-1,-1,0] * predictions_scaled[-1])
