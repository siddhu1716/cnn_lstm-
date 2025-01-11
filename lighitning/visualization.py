import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score, mean_squared_error, r2_score
from datetime import datetime
import optuna
from torch.utils.data import Dataset, DataLoader
import folium
from branca.colormap import LinearColormap
from scipy.interpolate import griddata
import requests
import elevation
import xarray as xr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import missingno as msno
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from branca.colormap import LinearColormap
import torch
from sklearn.preprocessing import StandardScaler
import joblib

def visualize_lightning_comparison(df, model_path):
    """
    Create visualizations comparing actual vs predicted lightning data
    and show locations on a map.
    """
    # 1. Load and prepare data
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%m/%d/%y %H:%M')

    # Extract required features
    df['Month'] = df['DateTime'].dt.month
    df['Hour'] = df['DateTime'].dt.hour
    df['DayOfYear'] = df['DateTime'].dt.dayofyear

    # Extract features and target
    X = df[['Latitude', 'Longitude', 'Month', 'Hour', 'DayOfYear']].values
    y_actual = df['Lightning_Data'].values

    # Scale the input features
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    # Load the saved model
    model = torch.load(model_path)
    model.eval()

    # Make predictions
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled)  # Convert to tensor
        # Reshape for sequence length compatibility (batch_size, seq_len, features)
        X_tensor = X_tensor.unsqueeze(0).repeat(5, 1, 1).permute(1, 0, 2)  # Adjust shape
        y_pred = model(X_tensor).squeeze().numpy()

    # 2. Create scatter plot comparison
    plt.figure(figsize=(15, 10))

    # Plot 1: Actual Lightning Data
    plt.subplot(2, 1, 1)
    scatter_actual = plt.scatter(df['Longitude'], df['Latitude'], 
                                  c=y_actual, cmap='viridis', 
                                  s=100, alpha=0.6)
    plt.colorbar(scatter_actual, label='Actual Lightning Intensity')
    plt.title('Actual Lightning Data')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Plot 2: Predicted Lightning Data
    plt.subplot(2, 1, 2)
    scatter_predicted = plt.scatter(df['Longitude'], df['Latitude'], 
                                     c=y_pred, cmap='plasma', 
                                     s=100, alpha=0.6)
    plt.colorbar(scatter_predicted, label='Predicted Lightning Intensity')
    plt.title('Predicted Lightning Data')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    plt.tight_layout()
    plt.show()

    # 3. Create interactive map
    center_lat = df['Latitude'].mean()
    center_lon = df['Longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=4)

    # Create color maps for actual data
    colormap = LinearColormap(
        colors=['blue', 'yellow', 'red'],
        vmin=min(y_actual),
        vmax=max(y_actual)
    )

    # Add points to the map
    for idx, row in df.iterrows():
        popup_text = f"""
        Latitude: {row['Latitude']:.2f}<br>
        Longitude: {row['Longitude']:.2f}<br>
        Actual Lightning: {y_actual[idx]:.2f}<br>
        Predicted Lightning: {y_pred[idx]:.2f}
        """
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=8,
            popup=popup_text,
            color=colormap(y_actual[idx]),
            fill=True,
            fillOpacity=0.7
        ).add_to(m)

    # Add colormap to the map
    colormap.add_to(m)
    colormap.caption = 'Lightning Intensity'

    # Save map to HTML file
    map_filename = 'lightning_map.html'
    m.save(map_filename)
    print(f"Interactive map saved as '{map_filename}'.")

    # 4. Calculate metrics
    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_actual, y_pred)

    metrics = {
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }

    return metrics

model_path = "/home/mcw/Desktop/work/enhanced_lightning_predictor.pth"
data_path = "/home/mcw/Desktop/work/lightning_data_region_daily.xlsx"

print("Loading data...")
df = pd.read_excel(data_path)
metrics = visualize_lightning_comparison(df, model_path)

print("\nModel Performance Metrics:")
print(f"MSE: {metrics['mse']:.4f}")
print(f"RMSE: {metrics['rmse']:.4f}")
print(f"R² Score: {metrics['r2']:.4f}")

print("\nVisualizations have been created:")
print("1. Matplotlib figures showing actual vs predicted lightning data")
print("2. Interactive map saved as 'lightning_map.html'")
