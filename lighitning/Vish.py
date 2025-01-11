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
from customized import EnhancedLightningPredictor,SpatioTemporalAttention

def load_pkl_model(model_path):
    from customized import EnhancedLightningPredictor
    with open(model_path, 'rb') as f:
        model = joblib.load(f)  # No globals argument needed
    return model

def visualize_lightning_comparison_pkl(df, model_path):
    """
    Visualize lightning data using a pre-trained .pkl model.
    """
    from customized import EnhancedLightningPredictor

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

    # Load the model
    model = load_pkl_model(model_path)
    model.eval()  # Ensure the model is in evaluation mode if it's PyTorch-based

    # Make predictions
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled)
        X_tensor = X_tensor.unsqueeze(1)  # Shape: [batch_size, seq_len, features]

        y_pred = model(X_tensor).squeeze().numpy()

    # 2. Visualization (same as before)
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

    # 3. Metrics calculation
    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_actual, y_pred)

    metrics = {
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }

    return metrics

# Usage
data_path = "/home/mcw/Desktop/work/lightning_data_region_daily.xlsx"

print("Loading data...")
df = pd.read_excel(data_path)
model_path = "/home/mcw/Desktop/work/LightningPredictor.pkl"
metrics = visualize_lightning_comparison_pkl(df, model_path)
print(metrics)
