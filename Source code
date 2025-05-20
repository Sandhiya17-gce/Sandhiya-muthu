
# CRACKING THE MARKET CODE WITH AI-DRIVEN STOCK PRICE PREDICTION USING TIME SERIES ANALYSIS

# Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
import math

# Step 1: Load the Data
df = yf.download("AAPL", start="2015-01-01", end="2023-12-31")
df = df[['Close']]
df.dropna(inplace=True)

# Step 2: Visualize the Data
plt.figure(figsize=(14,5))
plt.plot(df, label='Closing Price History')
plt.title("AAPL Stock Price History")
plt.xlabel("Date")
plt.ylabel("Close Price USD ($)")
plt.legend()
plt.show()

# Step 3: Normalize the Data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df)

# Step 4: Create Training and Test Sets
training_data_len = int(np.ceil( len(df) * .8 ))

train_data = scaled_data[0:int(training_data_len), :]
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Step 5: Build the LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Step 6: Compile and Train the Model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Step 7: Create Testing Data
test_data = scaled_data[training_data_len - 60: , :]
x_test = []
y_test = df[training_data_len:].values

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Step 8: Predict and Visualize the Results
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

rmse = math.sqrt(mean_squared_error(y_test, predictions))

train = df[:training_data_len]
valid = df[training_data_len:]
valid['Predictions'] = predictions

plt.figure(figsize=(14,6))
plt.title('Model Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Actual', 'Predicted'])
plt.show()

print(f"Root Mean Squared Error (RMSE): {rmse}")
