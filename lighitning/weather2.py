import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score, mean_squared_error, r2_score
from datetime import datetime
import optuna
from torch.utils.data import Dataset, DataLoader
import folium
from folium import plugins
from branca.colormap import LinearColormap

class LightningDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class LightningPredictor(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64, 32]):
        super(LightningPredictor, self).__init__()
        layers = []
        prev_size = input_size
        
        # Create dynamic layers based on hidden_sizes
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        # Final output layer
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def preprocess_data(df):
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    # Enhanced feature engineering
    df['Month'] = df['DateTime'].dt.month
    df['Hour'] = df['DateTime'].dt.hour
    df['DayOfYear'] = df['DateTime'].dt.dayofyear
    
    # Add cyclical encoding for temporal features
    df['Month_sin'] = np.sin(2 * np.pi * df['Month']/12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month']/12)
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour']/24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour']/24)
    
    features = ['Latitude', 'Longitude', 'Month_sin', 'Month_cos', 
                'Hour_sin', 'Hour_cos', 'DayOfYear']
    
    X = df[features].values
    y = df['Lightning_Data'].values
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    
    return X_scaled, y_scaled, scaler_X, scaler_y, features, df

def calculate_metrics(y_true, y_pred):
    # Convert continuous predictions to binary for classification metrics
    threshold = np.mean(y_true)  # You might want to adjust this threshold
    y_pred_binary = (y_pred > threshold).astype(int)
    y_true_binary = (y_true > threshold).astype(int)
    
    metrics = {
        'f1_score': f1_score(y_true_binary, y_pred_binary),
        'recall': recall_score(y_true_binary, y_pred_binary),
        'mse': mean_squared_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
    return metrics

def plot_predictions(df, y_true, y_pred, title):
    plt.figure(figsize=(15, 10))
    
    # Create scatter plot
    scatter = plt.scatter(df['Longitude'], df['Latitude'], 
                         c=y_true, cmap='viridis', 
                         label='Actual', alpha=0.6)
    plt.colorbar(scatter, label='Lightning Data (Actual)')
    
    # Overlay predictions
    scatter_pred = plt.scatter(df['Longitude'], df['Latitude'], 
                             c=y_pred, cmap='plasma', 
                             marker='x', s=100, 
                             label='Predicted', alpha=0.6)
    plt.colorbar(scatter_pred, label='Lightning Data (Predicted)')
    
    plt.title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid(True)
    plt.show()

def create_interactive_map(df, y_true, y_pred):
    # Create base map centered on data
    center_lat = df['Latitude'].mean()
    center_lon = df['Longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6)
    
    # Create colormaps for actual and predicted values
    colormap_actual = LinearColormap(
        colors=['blue', 'yellow', 'red'],
        vmin=min(y_true),
        vmax=max(y_true)
    )
    
    # Add actual data points
    for idx, row in df.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=6,
            color=colormap_actual(y_true[idx]),
            popup=f'Actual: {y_true[idx]:.4f}<br>Predicted: {y_pred[idx]:.4f}',
            fill=True
        ).add_to(m)
    
    # Add colormap to map
    colormap_actual.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def train_and_evaluate():
    # Load and preprocess data
    df = pd.read_excel('/home/mcw/Desktop/work/lightning_data_region_daily.xlsx')
    X_scaled, y_scaled, scaler_X, scaler_y, features, df_processed = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    
    # Initialize model with enhanced architecture
    model = LightningPredictor(input_size=len(features))
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Create data loaders
    train_dataset = LightningDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Train model
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(torch.FloatTensor(X_test)).numpy().ravel()
    
    # Inverse transform predictions
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    
    # Calculate metrics
    metrics = calculate_metrics(y_test_original, y_pred)
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Create visualizations
    plot_predictions(
        df.iloc[len(y_train):], 
        y_test_original, 
        y_pred, 
        "Lightning Data: Actual vs Predicted"
    )
    
    # Create interactive map
    interactive_map = create_interactive_map(
        df.iloc[len(y_train):], 
        y_test_original, 
        y_pred
    )
    interactive_map.save('lightning_prediction_map.html')

if __name__ == "__main__":
    train_and_evaluate()