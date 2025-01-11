import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

class TECPredictor(tf.keras.Model):
    def __init__(self, input_size, hidden_size=32, num_layers=2, dropout=0.3):
        super(TECPredictor, self).__init__()
        self.bidirectional_lstm = layers.Bidirectional(
            layers.LSTM(hidden_size, return_sequences=True, dropout=dropout, recurrent_dropout=dropout)
        )
        self.attention = layers.Attention()  # Use the Attention layer instead of Dense(softmax)
        self.lstm = layers.LSTM(hidden_size, return_sequences=False, dropout=dropout, recurrent_dropout=dropout)
        self.dense1 = layers.Dense(hidden_size // 2, activation='relu')
        self.dense2 = layers.Dense(1)

    def call(self, x):
        x = self.bidirectional_lstm(x)
        # Attention mechanism applied over time steps
        attention_weights = self.attention([x, x])  # Query and value are both x in self-attention
        x = x * attention_weights  # Element-wise multiplication of the LSTM output and attention weights
        x = tf.reduce_sum(x, axis=1)  # Aggregate along time axis
        x = self.lstm(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

def create_sequences(data, seq_length, features, target='TEC'):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[features].iloc[i:(i + seq_length)].values)
        y.append(data[target].iloc[i + seq_length])
    return np.array(X), np.array(y)

def train_model(model, train_dataset, val_dataset, optimizer, loss_fn, epochs):
    train_loss_results = []
    val_loss_results = []

    for epoch in range(epochs):
        train_loss = tf.keras.metrics.Mean()
        val_loss = tf.keras.metrics.Mean()

        # Training loop
        for x_batch, y_batch in train_dataset:
            with tf.GradientTape() as tape:
                predictions = model(x_batch, training=True)
                loss = loss_fn(y_batch, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss(loss)

        # Validation loop
        for x_batch, y_batch in val_dataset:
            val_predictions = model(x_batch, training=False)
            val_loss(loss_fn(y_batch, val_predictions))

        train_loss_results.append(train_loss.result())
        val_loss_results.append(val_loss.result())

        print(f"Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss.result().numpy()}, Val Loss: {val_loss.result().numpy()}")

    return train_loss_results, val_loss_results

def predict_and_evaluate(model, dataset, scaler):
    predictions = []
    actuals = []

    for x_batch, y_batch in dataset:
        preds = model(x_batch, training=False)
        predictions.extend(preds.numpy())
        actuals.extend(y_batch.numpy())

    predictions = scaler.inverse_transform(predictions)
    actuals = scaler.inverse_transform(actuals)

    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mape = mean_absolute_percentage_error(actuals, predictions)
    
    return predictions, actuals, rmse, mape

def main(df, train_end_date, seq_length=12, batch_size=16, epochs=2):
    # Feature engineering
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 365)

    # features = ['Ap', 'Kp', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    features = ['Ap', 'Kp', 'hour_sin', 'hour_cos']


    # Scaling
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    df[features] = feature_scaler.fit_transform(df[features])
    df['TEC'] = target_scaler.fit_transform(df[['TEC']])

    # Splitting data
    train = df[df['year'] <= train_end_date].copy()
    val_size = int(0.2 * len(train))
    train, val = train[:-val_size], train[-val_size:]
    test = df[df['year'] > train_end_date].copy()

    # Creating sequences
    X_train, y_train = create_sequences(train, seq_length, features)
    X_val, y_val = create_sequences(val, seq_length, features)
    X_test, y_test = create_sequences(test, seq_length, features)

    # Creating TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    # train_dataset = train_dataset.batch(batch_size).shuffle(len(X_train)).prefetch(tf.data.AUTOTUNE)
    # val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    # test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Initializing the model
    model = TECPredictor(input_size=len(features))
    # model = TECPredictor(input_size=len(features), hidden_size=128, num_layers=2)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.MeanSquaredError()

    # Training the model
    train_losses, val_losses = train_model(model, train_dataset, val_dataset, optimizer, loss_fn, epochs)

    # Evaluate on test set
    predictions, actuals, rmse, mape = predict_and_evaluate(model, test_dataset, target_scaler)

    print(f"Test RMSE: {rmse}, Test MAPE: {mape}")

    return model, predictions, actuals, train_losses, val_losses

# Reading the data
df = pd.read_excel("/home/mcw/Desktop/work/earthquake/DATAHyd.xlsx", header=None)

# Assign column names
df.columns = ['year', 'day', 'hour', 'Ap', 'Kp', 'TEC']

# Running the main function
# Try reducing seq_length
model, predictions, actuals, train_losses, val_losses = main(df, train_end_date=2009, seq_length=8)
