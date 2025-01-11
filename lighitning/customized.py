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
    and show locations on a map
    """
    df.loc[:, 'DateTime'] = pd.to_datetime(df['DateTime'], format='%m/%d/%y %H:%M')
    df.loc[:, 'Month'] = df['DateTime'].dt.month
    df.loc[:, 'Hour'] = df['DateTime'].dt.hour
    df.loc[:, 'DayOfYear'] = df['DateTime'].dt.dayofyear

    # Enhance features
    enhanced_df = enhance_features(df)
    feature_columns = [col for col in enhanced_df.columns if col != 'Lightning_Data']

    X = enhanced_df[feature_columns].values
    y_actual = enhanced_df['Lightning_Data'].values  # Use enhanced_df

    print("y_actual:", y_actual)
    print("Min:", y_actual.min(), "Max:", y_actual.max())

    y_actual = np.nan_to_num(y_actual, nan=0.0)  # Replace NaN with 0
    y_actual = np.clip(y_actual, 0, 100)  # Adjust the range based on your data

    # Load the saved scaler and model
    scaler_X = joblib.load("/home/mcw/Desktop/work/scaler_X.pkl")
    # scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    model = joblib.load(model_path)
    
    # Use transform instead of fit_transform
    X_scaled = scaler_X.transform(X)

    # Make predictions
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled)
        sequence_length = 5
        X_tensor = X_tensor.unsqueeze(1).repeat(1, sequence_length, 1)
        
        batch_size = 1024
        y_pred = []
        
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            pred = model(batch).squeeze().numpy()
            y_pred.extend(pred)
            
        y_pred = np.array(y_pred)
        error_analysis = pd.DataFrame({'Actual': y_actual, 'Predicted': y_pred})
        print(error_analysis.describe())
    # 2. Create scatter plot comparison
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Actual vs Predicted Lightning Data
    plt.subplot(2, 1, 1)
    scatter = plt.scatter(df['Longitude'], df['Latitude'], 
                         c=y_actual, cmap='viridis', 
                         s=100, alpha=0.6)
    plt.colorbar(scatter, label='Actual Lightning Intensity')
    plt.title('Actual Lightning Data')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # Plot 2: Predicted Lightning Data
    plt.subplot(2, 1, 2)
    scatter = plt.scatter(df['Longitude'], df['Latitude'], 
                         c=y_pred, cmap='viridis', 
                         s=100, alpha=0.6)
    plt.colorbar(scatter, label='Predicted Lightning Intensity')
    plt.title('Predicted Lightning Data')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    plt.tight_layout()
    plt.savefig("comparison_plot.png")
    
    # 3. Create interactive map
    center_lat = df['Latitude'].mean()
    center_lon = df['Longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=4)
    
    # Create color maps for actual and predicted data
    colormap = LinearColormap(
        colors=['blue', 'yellow', 'red'],
        vmin=min(y_actual),
        vmax=max(y_actual)
    )
    
    # Add points to map
    for idx in range(len(df)):
        # Create popup text
        popup_text = f"""
        Latitude: {df['Latitude'].iloc[idx]:.2f}<br>
        Longitude: {df['Longitude'].iloc[idx]:.2f}<br>
        Actual Lightning: {y_actual[idx]:.2f}<br>
        Predicted Lightning: {y_pred[idx]:.2f}
        """
        
        # Add marker with color based on lightning intensity
        folium.CircleMarker(
            location=[df['Latitude'].iloc[idx], df['Longitude'].iloc[idx]],
            radius=8,
            popup=popup_text,
            color=colormap(y_actual[idx]),
            fill=True,
            fillOpacity=0.7
        ).add_to(m)
    
    # Add colormap to map
    colormap.add_to(m)
    colormap.caption = 'Lightning Intensity'
    
    # Save map to HTML file
    m.save('lightning_map.html')
    
    # 4. Calculate and return metrics
    metrics = {
        'mse': np.mean((y_actual - y_pred) ** 2),
        'rmse': np.sqrt(np.mean((y_actual - y_pred) ** 2)),
        'r2': 1 - np.sum((y_actual - y_pred) ** 2) / np.sum((y_actual - np.mean(y_actual)) ** 2)
    }
    
    return metrics


class MissingValueHandler:
    def __init__(self, strategy='advanced'):
        self.strategy = strategy
        self.imputers = {}
        self.statistics = {}
    
    def analyze_missing_values(self, df):
        """Analyze patterns of missing values in the dataset"""
        # Calculate missing value statistics
        missing_stats = {
            'total_missing': df.isnull().sum().sum(),
            'missing_by_column': df.isnull().sum(),
            'missing_percentage': (df.isnull().sum() / len(df)) * 100,
            'rows_with_missing': df.isnull().any(axis=1).sum(),
            'rows_with_missing_percentage': (df.isnull().any(axis=1).sum() / len(df)) * 100
        }
        
        # Visualize missing value patterns
        plt.figure(figsize=(12, 6))
        msno.matrix(df)
        plt.title("Missing Value Patterns")
        plt.show()
        
        plt.figure(figsize=(12, 6))
        msno.heatmap(df)
        plt.title("Missing Value Correlation Heatmap")
        plt.show()
        
        return missing_stats
    
    def handle_missing_values(self, df, temporal_columns=None):
        """
        Handle missing values using various strategies
        
        Parameters:
        df: pandas DataFrame
        temporal_columns: list of column names that represent temporal features
        """
        df_cleaned = df.copy()
        
        # Handle temporal features separately if specified
        if temporal_columns:
            temporal_data = df_cleaned[temporal_columns]
            non_temporal_data = df_cleaned.drop(columns=temporal_columns)
        else:
            temporal_data = pd.DataFrame()
            non_temporal_data = df_cleaned
        
        if self.strategy == 'advanced':
            # Use different imputation strategies for different types of columns
            
            # Numeric columns
            numeric_columns = non_temporal_data.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_columns) > 0:
                self.imputers['numeric'] = IterativeImputer(
                    max_iter=10, 
                    random_state=42,
                    initial_strategy='median'
                )
                numeric_imputed = self.imputers['numeric'].fit_transform(
                    non_temporal_data[numeric_columns]
                )
                non_temporal_data[numeric_columns] = numeric_imputed
            
            # Categorical columns
            categorical_columns = non_temporal_data.select_dtypes(include=['object', 'category']).columns
            if len(categorical_columns) > 0:
                self.imputers['categorical'] = SimpleImputer(
                    strategy='most_frequent'
                )
                categorical_imputed = self.imputers['categorical'].fit_transform(
                    non_temporal_data[categorical_columns]
                )
                non_temporal_data[categorical_columns] = categorical_imputed
            
            # Spatial columns (Latitude, Longitude)
            spatial_columns = [col for col in numeric_columns if col in ['Latitude', 'Longitude']]
            if spatial_columns:
                self.imputers['spatial'] = KNNImputer(n_neighbors=5)
                spatial_imputed = self.imputers['spatial'].fit_transform(
                    non_temporal_data[spatial_columns]
                )
                non_temporal_data[spatial_columns] = spatial_imputed
            
        else:  # simple strategy
            self.imputers['simple'] = SimpleImputer(strategy='median')
            non_temporal_data = pd.DataFrame(
                self.imputers['simple'].fit_transform(non_temporal_data),
                columns=non_temporal_data.columns
            )
        
        # Handle temporal features if present
        if not temporal_data.empty:
            self.imputers['temporal'] = KNNImputer(n_neighbors=5)
            temporal_imputed = self.imputers['temporal'].fit_transform(temporal_data)
            temporal_data = pd.DataFrame(temporal_imputed, columns=temporal_columns)
            
            # Combine temporal and non-temporal data
            df_cleaned = pd.concat([temporal_data, non_temporal_data], axis=1)
        else:
            df_cleaned = non_temporal_data
        
        return df_cleaned
    
    def validate_imputation(self, original_df, imputed_df):
        """Validate the imputation results"""
        validation_results = {
            'columns_with_missing_before': original_df.isnull().sum(),
            'columns_with_missing_after': imputed_df.isnull().sum(),
            'total_missing_before': original_df.isnull().sum().sum(),
            'total_missing_after': imputed_df.isnull().sum().sum()
        }
        
        # Compare distributions before and after imputation
        for column in original_df.select_dtypes(include=['float64', 'int64']).columns:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            original_df[column].hist(alpha=0.5, label='Original', bins=30)
            imputed_df[column].hist(alpha=0.5, label='Imputed', bins=30)
            plt.title(f'Distribution Comparison - {column}')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            sns.boxplot(data=pd.DataFrame({
                'Original': original_df[column],
                'Imputed': imputed_df[column]
            }))
            plt.title(f'Boxplot Comparison - {column}')
            
            plt.tight_layout()
            plt.show()
        
        return validation_results

def preprocess_data_with_missing_handling(df):
    """Main preprocessing function with missing value handling"""
    # Initialize missing value handler
    mv_handler = MissingValueHandler(strategy='advanced')
    
    # Analyze missing values
    print("Analyzing missing values...")
    missing_stats = mv_handler.analyze_missing_values(df)
    print("\nMissing Value Statistics:")
    for key, value in missing_stats.items():
        print(f"\n{key}:")
        print(value)
    
    # Handle missing values
    print("\nHandling missing values...")
    temporal_columns = ['DateTime']
    df_cleaned = mv_handler.handle_missing_values(df, temporal_columns)
    
    # Validate imputation
    print("\nValidating imputation results...")
    validation_results = mv_handler.validate_imputation(df, df_cleaned)
    print("\nImputation Validation Results:")
    for key, value in validation_results.items():
        print(f"\n{key}:")
        print(value)
    
    # Continue with regular preprocessing
    df_cleaned['DateTime'] = pd.to_datetime(df_cleaned['DateTime'])
    
    # Enhanced feature engineering (keeping the same as before)
    df_cleaned['Month'] = df_cleaned['DateTime'].dt.month
    df_cleaned['Hour'] = df_cleaned['DateTime'].dt.hour
    df_cleaned['DayOfYear'] = df_cleaned['DateTime'].dt.dayofyear
    
    features = ['Latitude', 'Longitude', 'Month', 'Hour', 'DayOfYear']
    
    X = df_cleaned[features].values
    y = df_cleaned['Lightning_Data'].values
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    
    return X_scaled, y_scaled, scaler_X, scaler_y, features, df_cleaned

class EnhancedLightningDataset(Dataset):
    def __init__(self, features, targets, sequence_length=5):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.features) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        # Return sequence of data points
        return (self.features[idx:idx + self.sequence_length], 
                self.targets[idx + self.sequence_length - 1])

class SpatioTemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4)
        
    def forward(self, x):
        x = x.transpose(0, 1)  # Reshape for attention
        attn_output, _ = self.attention(x, x, x)
        return attn_output.transpose(0, 1)
    
class EnhancedLightningPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        
        # Modify spatial_net to match input dimensions
        self.spatial_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # First layer matches input size
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout)
        )
        
        # Rest remains the same
        self.lstm = nn.LSTM(
            hidden_size, 
            hidden_size, 
            num_layers=num_layers, 
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.attention = SpatioTemporalAttention(hidden_size)
        
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        
        # Reshape input for spatial network
        x_reshaped = x.reshape(-1, features)  # Flatten batch and sequence dimensions
        spatial_out = self.spatial_net(x_reshaped)  # Process through spatial network
        spatial_features = spatial_out.reshape(batch_size, seq_len, -1)  # Reshape back
        
        # Process temporal sequence
        lstm_out, _ = self.lstm(spatial_features)
        
        # Apply attention
        attended = self.attention(lstm_out)
        
        # Take final timestep for prediction
        final_features = attended[:, -1, :]
        
        # Generate prediction
        return self.output_net(final_features)
    

def fetch_environmental_data(lat, lon, date):
    """
    Simulate fetching environmental data for the given coordinates and date.
    In a real implementation, this would call weather APIs or read from a database.
    """
    # Create synthetic environmental features based on location and time
    hour = pd.Timestamp(date).hour
    month = pd.Timestamp(date).month
    
    # Simulate temperature based on latitude and time of day
    base_temp = 25 - (abs(lat - 23.5) * 0.5)  # Warmer near equator
    time_effect = -np.cos(hour * 2 * np.pi / 24) * 5  # Daily temperature cycle
    season_effect = np.cos((month - 6) * 2 * np.pi / 12) * 10  # Seasonal variation
    temperature = base_temp + time_effect + season_effect
    
    # Simulate humidity based on longitude (distance from coast) and temperature
    base_humidity = 70 - (abs(lon % 90 - 45) * 0.5)
    humidity = base_humidity - (temperature - 20) * 0.5
    humidity = np.clip(humidity, 30, 100)
    
    # Simulate pressure based on elevation (estimated from latitude)
    base_pressure = 1013.25 - (abs(lat) * 0.1)
    pressure = base_pressure + np.random.normal(0, 1)
    
    return {
        'temperature': temperature,
        'humidity': humidity,
        'pressure': pressure
    }

def enhance_features(df):
    """Add derived and environmental features to the dataset"""
    enhanced_data = []
    
    for _, row in df.iterrows():
        # Get environmental data
        env_data = fetch_environmental_data(row['Latitude'], row['Longitude'], row['DateTime'])
        
        # Calculate derived features
        distance_from_equator = abs(row['Latitude'])
        is_land = True if (row['Longitude'] % 90) > 20 else False  # Simplified land/water detection
        
        # Combine all features
        features = {
            'Latitude': row['Latitude'],
            'Longitude': row['Longitude'],
            'Month': row['DateTime'].month,
            'Hour': row['DateTime'].hour,
            'DayOfYear': row['DateTime'].dayofyear,
            'Distance_From_Equator': distance_from_equator,
            'Is_Land': int(is_land),
            'Temperature': env_data['temperature'],
            'Humidity': env_data['humidity'],
            'Pressure': env_data['pressure'],
            'Lightning_Data': row['Lightning_Data']
        }
        
        enhanced_data.append(features)
    
    return pd.DataFrame(enhanced_data)

def train_with_cross_validation(model, X, y, num_folds=5):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train model
        train_dataset = EnhancedLightningDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        criterion = nn.MSELoss()
        
        for epoch in range(50):  # Reduced epochs for example
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            val_dataset = EnhancedLightningDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=32)
            val_predictions = []
            val_targets = []
            
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X).squeeze()
                val_predictions.extend(outputs.numpy())
                val_targets.extend(batch_y.numpy())
            
            score = r2_score(val_targets, val_predictions)
            scores.append(score)
            
        print(f"Fold {fold + 1} R² Score: {score:.4f}")
    
    return np.mean(scores), np.std(scores)

def analyze_feature_importance(X, y):
    """Analyze feature importance using Random Forest"""
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return feature_importance


if __name__ == "__main__":
    try:
        # # Load data
        # print("Loading data...")
        df = pd.read_excel('/home/mcw/Desktop/work/lightning_data_region_daily.xlsx')

        df['DateTime'] = pd.to_datetime(df['DateTime'], format='%m/%d/%y %H:%M')
        df = df.sort_values('DateTime')  

        # Use 80% for training, 20% for testing
        train_size = int(len(df) * 0.8)
        train_df = df[:train_size]
        test_df = df[train_size:]

        # Now process training data
        print("Handling missing values for training data...")
        X_scaled_train, y_scaled_train, scaler_X, scaler_y, features, train_df_cleaned = preprocess_data_with_missing_handling(train_df)

        
        print("Enhancing features for training data...")
        enhanced_train_df = enhance_features(train_df_cleaned)
        # Train the model
        X_train = enhanced_train_df.drop('Lightning_Data', axis=1).values
        y_train = enhanced_train_df['Lightning_Data'].values
                
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        # X_scaled_train = scaler_X.fit_transform(X_train)
        # y_scaled_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

        # # Initialize model with correct input size
        # input_size = X_train.shape[1]  # This will be larger than 5 due to enhanced features
        # model = EnhancedLightningPredictor(input_size=input_size)

        # # Train and evaluate model
        # print("\nTraining model...")
        # mean_score, std_score = train_with_cross_validation(model, X_scaled_train, y_scaled_train)
        # print(f"Cross-Validation Mean R² Score: {mean_score:.4f}")
        # print(f"Cross-Validation Std Dev R² Score: {std_score:.4f}")

        model_path="/home/mcw/Desktop/work/LightningPredictor.pkl"
        # joblib.dump(model,model_path)
        # joblib.dump(scaler_X, "/home/mcw/Desktop/work/scaler_X.pkl")
        # joblib.dump(scaler_y, "/home/mcw/Desktop/work/scaler_y.pkl")
        # # scaler_X = joblib.load("/home/mcw/Desktop/work/scaler_X.pkl")
        # print("Evaluating on test set...")
        # Process test data
        test_df_cleaned = preprocess_data_with_missing_handling(test_df)[5]  # Get cleaned df
        enhanced_test_df = enhance_features(test_df_cleaned)

        # Prepare test features
        X_test = enhanced_test_df.drop('Lightning_Data', axis=1).values
        y_test = enhanced_test_df['Lightning_Data'].values
        X_scaled_test = scaler_X.transform(X_test)  # Use transform, not fit_transform

        # Test visualization
        metrics = visualize_lightning_comparison(test_df,model_path)

        print("\nModel Performance Metrics:")
        print(f"MSE: {metrics['mse']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"R² Score: {metrics['r2']:.4f}")
        
        print("\nVisualizations have been created:")
        print("1. Matplotlib figures showing actual vs predicted lightning data")
        print("2. Interactive map saved as 'lightning_map.html'")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise   

# df = pd.read_excel('/home/mcw/Desktop/work/lightning_data_region_daily.xlsx')
# model_path = "/home/mcw/Desktop/work/LightningPredictor.pkl"
# metrics = visualize_lightning_comparison(df, model_path)
        
# print("\nModel Performance Metrics:")
# print(f"MSE: {metrics['mse']:.4f}")
# print(f"RMSE: {metrics['rmse']:.4f}")
# print(f"R² Score: {metrics['r2']:.4f}")

# print("\nVisualizations have been created:")
# print("1. Matplotlib figures showing actual vs predicted lightning data")
# print("2. Interactive map saved as 'lightning_map.html'")

        
