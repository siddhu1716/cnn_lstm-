import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd


def create_cyclical_features(df):
    """Create cyclical features for temporal columns"""
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['day_sin'] = np.sin(2 * np.pi * df['day']/365)
    df['day_cos'] = np.cos(2 * np.pi * df['day']/365)
    return df

def prepare_sequences(data, seq_size, features, target='TEC'):
    """Create sequences for LSTM input"""
    X, y = [], []
    for i in range(len(data) - seq_size):
        X.append(data[features].iloc[i:(i + seq_size)].values)
        y.append(data[target].iloc[i + seq_size])
    return np.array(X), np.array(y)

def create_model(seq_size, n_features):
    """Create LSTM model with modified autoencoder architecture"""
    model = Sequential([
        LSTM(128, input_shape=(seq_size, n_features), return_sequences=True),
        Dropout(0.3),
        LSTM(128),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_tec_model(df, train_end_date, seq_size=24):
    """Train and evaluate TEC prediction model"""
    # Preprocess data
    df = create_cyclical_features(df)
    
    # Define features
    features = ['Ap', 'Kp', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'TEC']

    
    # Scale features
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    
    # Split data
    train = df[df['year'] <= train_end_date].copy()
    test = df[df['year'] > train_end_date].copy()
    
    # Create sequences
    X_train, y_train = prepare_sequences(train, seq_size, features)
    X_test, y_test = prepare_sequences(test, seq_size, features)
    
    # Create and train model
    model = create_model(seq_size, len(features))
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, 
                                 restore_best_weights=True)
    
    history = model.fit(X_train, y_train, 
                       validation_split=0.2,
                       epochs=2, 
                       batch_size=16,
                       callbacks=[early_stopping],
                       verbose=1)
    
    # Make predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Inverse transform predictions
    tec_scaler = MinMaxScaler()
    tec_scaler.fit(df[['TEC']])
    train_pred = tec_scaler.inverse_transform(train_pred)
    test_pred = tec_scaler.inverse_transform(test_pred)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(
        tec_scaler.inverse_transform(y_train.reshape(-1, 1)), train_pred))
    test_rmse = np.sqrt(mean_squared_error(
        tec_scaler.inverse_transform(y_test.reshape(-1, 1)), test_pred))
    test_mape = mean_absolute_percentage_error(
        tec_scaler.inverse_transform(y_test.reshape(-1, 1)), test_pred)
    
    return model, history, train_rmse, test_rmse, test_mape,features

def plot_results(history, y_true, y_pred, title):
    """Plot training history and predictions"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot training history
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Training History')
    ax1.legend()
    
    # Plot predictions vs actual
    ax2.plot(y_true, label='Actual')
    ax2.plot(y_pred, label='Predicted')
    ax2.set_title(title)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


# Load the dataset
df = pd.read_excel('/home/mcw/Desktop/work/earthquake/DATAHyd.xlsx')
df.columns = ['year', 'day', 'hour', 'Ap', 'Kp', 'TEC']

train_end_date = 2009
seq_size = 12
model, history, train_rmse, test_rmse, test_mape, features = train_tec_model(df, train_end_date, seq_size)
print(f"Training RMSE: {train_rmse}")
print(f"Testing RMSE: {test_rmse}")
print(f"Testing MAPE: {test_mape}")
# Generate predictions for testing set
y_true = df[df['year'] > train_end_date]['TEC'].iloc[seq_size:].values
y_pred = model.predict(prepare_sequences(df[df['year'] > train_end_date], seq_size, features)[0])

plot_results(history, y_true, y_pred, title="Predicted vs Actual TEC")
model.save('tec_prediction_model.h5')


