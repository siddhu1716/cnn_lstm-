# -*- coding: utf-8 -*-
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, RepeatVector, TimeDistributed
from keras.callbacks import EarlyStopping
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Load data
#dataframe = pd.read_csv(r'C:\Users\sshas\Downloads\GE.csv')
dataframe = pd.read_csv('/home/mcw/Desktop/work/earthquake/GE.csv')

df = dataframe[['Date', 'Close']]
df['Date'] = pd.to_datetime(df['Date'])

# Print available date range
print("Available date range in the dataset:")
print("Start date is: ", df['Date'].min())
print("End date is: ", df['Date'].max())

# Define training start date and sequence size
train_start_date = '2023-10-01'
seq_size = 10
min_seq_size = 1

# Convert train_start_date to Timestamp
train_start_date = pd.to_datetime(train_start_date)

# Filter data to ensure enough training data
train = df[df['Date'] < train_start_date].copy()
test = df[df['Date'] >= train_start_date].copy()

# Ensure enough data to create sequences with the initial sequence size
if len(train) < seq_size or len(test) < seq_size:
    raise ValueError("Not enough data to create sequences with the initial sequence size. Please provide more data or adjust the sequence size.")

# Function to create sequences
def to_sequences(x, y, seq_size=1):
    x_values = []
    y_values = []

    for i in range(len(x) - seq_size):
        x_values.append(x.iloc[i:(i + seq_size)].values)
        y_values.append(y.iloc[i + seq_size])

    return np.array(x_values), np.array(y_values)

# Ensure there is enough data for sequence creation
def check_and_adjust_seq_size(train, test, seq_size, min_seq_size):
    while (len(train) < seq_size or len(test) < seq_size) and seq_size > min_seq_size:
        print("Not enough data for the given sequence size and date range. Adjusting sequence size.")
        seq_size -= 1
    if seq_size < min_seq_size:
        raise ValueError("Not enough data to create sequences even with the minimum sequence size. Please check the dataset and date range.")
    return seq_size

seq_size = check_and_adjust_seq_size(train, test, seq_size, min_seq_size)
print(f"Adjusted sequence size: {seq_size}")

# Normalize data
scaler = StandardScaler()
train['Close'] = scaler.fit_transform(train[['Close']])
test['Close'] = scaler.transform(test[['Close']])

# Create sequences
trainX, trainY = to_sequences(train[['Close']], train['Close'], seq_size)
testX, testY = to_sequences(test[['Close']], test['Close'], seq_size)

# Ensure there are enough sequences
if trainX.shape[0] == 0 or testX.shape[0] == 0:
    raise ValueError("Not enough data to create sequences. Consider reducing the sequence size or ensuring sufficient data.")

# Define model
model = Sequential()
model.add(LSTM(128, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dropout(rate=0.2))
model.add(RepeatVector(trainX.shape[1]))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(rate=0.2))
model.add(TimeDistributed(Dense(trainX.shape[2])))
model.compile(optimizer='adam', loss='mae')
model.summary()

# Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Fit model
history = model.fit(trainX, trainY, epochs=100, batch_size=32, validation_split=0.1, callbacks=[early_stopping], verbose=1)

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

# Evaluate model
test_loss = model.evaluate(testX, testY)
print("Test Loss:", test_loss)

# Add monthly high points
test['Month'] = test['Date'].dt.to_period('M')
monthly_highs = test.groupby('Month')['Close'].idxmax()

monthly_highs_points = test.loc[monthly_highs].sort_values(by='Date')

# Plot anomalies with monthly high points
plt.figure(figsize=(10, 6))
sns.lineplot(x=test['Date'], y=scaler.inverse_transform(test[['Close']]).flatten(), label='Close')

# Highlight monthly highs in red
plt.scatter(monthly_highs_points['Date'], scaler.inverse_transform(monthly_highs_points[['Close']]), color='r', label='Monthly Highs')

plt.legend()
plt.show()
