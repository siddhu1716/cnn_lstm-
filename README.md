# cnn_lstm

## Earthquke TEC Prediction using LSTM Autoencoder

## Lighitning Prediction using Conv LSTM 

## Overview
This project implements a deep learning model to predict Total Electron Content (TEC) values using an LSTM-based architecture. The model utilizes geomagnetic indices (Ap, Kp) and temporal features to forecast TEC values, which can be valuable for studying ionospheric behavior and potentially identifying earthquake precursors.

## Features
- Bidirectional LSTM with attention mechanism
- Separate scaling for features and target variables
- Advanced preprocessing including cyclical encoding of temporal features
- Comprehensive visualization tools for model evaluation
- Model saving and loading functionality
- Early stopping and learning rate scheduling
- Gradient clipping for stable training

## Prerequisites
```
python >= 3.8
torch
numpy
pandas
scikit-learn
matplotlib
seaborn
```

## Installation
1. Clone the repository:
```bash
git clone [https://github.com/siddhu1716/cnn_lstm-.git]
cd lighitning
cd earth quake
```

2. Install required packages:
```bash
pip install -r requirements.txt
```


## Project Structure
```
├── earthquake/
│   ├── DATAHyd.xlsx
│   ├── GE.csv        # TECDataset class
│   ├── LSTMAUTOENCODER.py      # TECPredictor model
│   └── earth.py , earth_tensorflow.py , earth_pytorch.py        # each noteboook represents a different straregy and librariy
├── lighitning/
│   ├── preprocessing.py  # Data preparation
│   └── visualization.py  # Plotting functions
├── main.py              # Main training script
└── README.md
```

## Data Format
The input data should be an Excel file with the following columns:
- year: Year of observation
- day: Day of year (1-365)
- hour: Hour of day (0-23)
- Ap: Geomagnetic Ap index
- Kp: Geomagnetic Kp index
- TEC: Total Electron Content (target variable)

## Model Architecture
- Input layer: Processes sequence of features (Ap, Kp, temporal encodings)
- Bidirectional LSTM layers with dropout
- Attention mechanism for focusing on relevant time steps
- Dense layers for final prediction
- Layer normalization for training stability

## Usage
1. Prepare your data in the required format
2. Run the training script:


## Model Parameters
- `seq_length`: Length of input sequences (default: 24)
- `batch_size`: Training batch size (default: 32)
- `hidden_size`: LSTM hidden layer size (default: 256)
- `num_layers`: Number of LSTM layers (default: 3)
- `dropout`: Dropout rate (default: 0.3)

## Visualization
The project includes several visualization tools:
1. Training history plots
2. Actual vs predicted values comparison
3. Residual analysis
4. Error distribution plots

## Model Saving/Loading
```python
# Save model
model_manager.save('tec_model.pth')

# Load model
loaded_manager = ModelManager.load('tec_model.pth')
```

## Performance Metrics
The model's performance is evaluated using:
- Root Mean Square Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

## Contributing
Feel free to submit issues, fork the repository, and create pull requests for any improvements.


## Authors
[@shivanampalli]


## Contact
[shivanampalli@gmail.com]
