import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import optuna
from torch.utils.data import Dataset, DataLoader

# Custom Dataset
class LightningDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Neural Network Model
class LightningPredictor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LightningPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size//2, 1)
        )
    
    def forward(self, x):
        return self.network(x)

# Data Preprocessing
def preprocess_data(df):
    # Convert DateTime to features
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['Month'] = df['DateTime'].dt.month
    df['Hour'] = df['DateTime'].dt.hour
    df['DayOfYear'] = df['DateTime'].dt.dayofyear
    
    # Create feature matrix
    features = ['Latitude', 'Longitude', 'Month', 'Hour', 'DayOfYear']
    X = df[features].values
    y = df['Lightning_Data'].values
    
    # Scale features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    
    return X_scaled, y_scaled, scaler_X, scaler_y, features

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X).squeeze()
                val_loss += criterion(outputs, batch_y).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.4f}, Val Loss = {val_loss/len(val_loader):.4f}')

# Optuna objective for hyperparameter optimization
def objective(trial):
    # Hyperparameters to optimize
    hidden_size = trial.suggest_int('hidden_size', 32, 256)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_int('batch_size', 16, 128)
    
    # Create model with trial parameters
    model = LightningPredictor(input_size=len(features), hidden_size=hidden_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Train for a few epochs
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50)
    
    # Evaluate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X).squeeze()
            val_loss += criterion(outputs, batch_y).item()
    
    return val_loss / len(val_loader)

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    df = pd.read_excel('lightning_data.xls')
    X_scaled, y_scaled, scaler_X, scaler_y, features = preprocess_data(df)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = LightningDataset(X_train, y_train)
    val_dataset = LightningDataset(X_val, y_val)
    
    # Hyperparameter optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    
    # Get best parameters
    best_params = study.best_params
    print("Best hyperparameters:", best_params)
    
    # Train final model with best parameters
    final_model = LightningPredictor(input_size=len(features), hidden_size=best_params['hidden_size'])
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['learning_rate'])
    final_criterion = nn.MSELoss()
    
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'])
    
    train_model(final_model, train_loader, val_loader, final_criterion, final_optimizer, num_epochs=200)