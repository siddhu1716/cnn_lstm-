import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import gc

class TECDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class TECPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class ModelManager:
    def __init__(self, model, scaler, seq_length, features):
        self.model = model
        self.scaler = scaler
        self.seq_length = seq_length
        self.features = features

    def save(self, path):
        save_dict = {
            'model_state': self.model.state_dict(),
            'scaler': self.scaler,
            'seq_length': self.seq_length,
            'features': self.features,
            'model_config': {
                'input_size': len(self.features),
                'hidden_size': 128  # Match the model's hidden size
            }
        }
        torch.save(save_dict, path)

    @classmethod
    def load(cls, path, device='cuda'):
        save_dict = torch.load(path, map_location=device)
        model = TECPredictor(**save_dict['model_config']).to(device)
        model.load_state_dict(save_dict['model_state'])
        return cls(model, save_dict['scaler'], save_dict['seq_length'], save_dict['features'])

def create_sequences(data, seq_length, features, target='TEC'):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[features].iloc[i:(i + seq_length)].values)
        y.append(data[target].iloc[i + seq_length])
    return np.array(X), np.array(y)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            # Clear cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y.unsqueeze(1)).item()
        
        # Calculate average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        # Force garbage collection
        gc.collect()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses

def predict_and_evaluate(model, data_loader, target_scaler, device):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.numpy())
    
    predictions = target_scaler.inverse_transform(np.array(predictions))
    actuals = target_scaler.inverse_transform(np.array(actuals))
    
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mape = mean_absolute_percentage_error(actuals, predictions)
    
    return predictions, actuals, rmse, mape
    
def main(df, train_end_date, seq_length=12, batch_size=16):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True  # Optimize CUDA operations
    
    # Basic feature engineering
    features = ['Ap', 'Kp']
    
    # Scale data
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    
    # Split data
    train = df[df['year'] <= train_end_date].copy()
    val_size = int(0.2 * len(train))
    train, val = train[:-val_size], train[-val_size:]
    test = df[df['year'] > train_end_date].copy()
    
    # Create sequences
    X_train, y_train = create_sequences(train, seq_length, features)
    X_val, y_val = create_sequences(val, seq_length, features)
    X_test, y_test = create_sequences(test, seq_length, features)
    
    # Create datasets and dataloaders
    train_dataset = TECDataset(X_train, y_train)
    val_dataset = TECDataset(X_val, y_val)
    test_dataset = TECDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    model = TECPredictor(input_size=len(features)).to(device)
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Train model
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        scheduler, num_epochs=50, device=device
    )
    
    # Create model manager and save
    model_manager = ModelManager(model, scaler, seq_length, features)
    model_manager.save('tec_model.pth')
    
    # Load best model and evaluate
    model.load_state_dict(torch.load('best_model.pth'))
    predictions, actuals, rmse, mape = predict_and_evaluate(model, test_loader, scaler, device)
    
    return model_manager, predictions, actuals, rmse, mape, train_losses, val_losses

if __name__ == "__main__":
    # Load data
    df = pd.read_excel("/home/mcw/Desktop/work/earthquake/DATAHyd.xlsx", header=None)
    df.columns = ['year', 'day', 'hour', 'Ap', 'Kp', 'TEC']
    
    # Run main function
    manager, predictions, actuals, rmse, mape, train_losses, val_losses = main(df, train_end_date=2009)
    
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.4f}")