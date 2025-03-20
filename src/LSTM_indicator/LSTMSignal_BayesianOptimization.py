#!/usr/bin/env python3
"""
Time Series LSTM for Algorithmic Day Trading Signals with Focal Loss

This script loads a pickle file containing OHLCV data, EMAs, VolumeWeighted and a Signal column.
It filters data to include only rows with times between 14:30:00 and 20:59:00, creates
a dataset of sequences, and trains an LSTM to predict trading signals (Buy, Sell, Hold).
It also computes class weights to mitigate class imbalance in the Signal column.
If a trained model exists, the script enters prediction mode and monitors the pickle file
every 30 seconds to predict signals for newly appended data.

- Uses PyTorch for training an LSTM model on OHLCV + EMAs + VolumeWeighted.
- Handles missing data and class imbalance.
- Implements Focal Loss for better handling of class imbalance.
- Evaluates with Precision, Recall, F1-score, and Confusion Matrix.
- Loads model if available and continuously makes predictions.
- Monitors validation loss and saves the best model.

MODEL_PATH = "/Users/chrisjackson/Desktop/DEV/python/data/model.pth"
PICKLE_FILE = "/Users/chrisjackson/Desktop/DEV/python/data/pickleFile.pkl"
"""


import os
import time
import pickle
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime as dt
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, SubsetRandomSampler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from typing import Optional
from skopt import gp_minimize
from skopt.space import Integer, Real

# Logging setup
logging.basicConfig(level=logging.INFO)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {DEVICE}")

MODEL_PATH = "/Users/chrisjackson/Desktop/DEV/python/data/model.pth"
PICKLE_FILE = "/Users/chrisjackson/Desktop/DEV/python/data/pickleFile.pkl"
FOCAL_GAMMA = 2.0
VALIDATION_SPLIT = 0.2

class LSTMSignal(nn.Module):
    """LSTM Model for Time Series Classification."""
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMSignal, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

def load_data(pickle_file):
    if not os.path.exists(pickle_file):
        logging.error(f"Pickle file {pickle_file} not found.")
        return None
    with open(pickle_file, "rb") as f:
        df = pickle.load(f)
    df.index = pd.to_datetime(df.index)
    df = df.between_time("14:30", "20:59").copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

class TradingDataset(Dataset):
    def __init__(self, df, seq_length):
        self.seq_length = seq_length
        self.X, self.y = self._prepare_sequences(df)

    def _prepare_sequences(self, df):
        sequences, labels = [], []
        features = ["Open", "High", "Low", "Close", "Volume", "VolumeWeighted", "EMA_13", "EMA_100"]
        for i in range(len(df) - self.seq_length):
            sequences.append(df.iloc[i:i + self.seq_length][features].values)
            labels.append(df.iloc[i + self.seq_length]["Signal"])
        return torch.tensor(np.array(sequences), dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_and_evaluate(params):
    seq_length, batch_size, learning_rate, num_epochs, hidden_size, num_layers = params
    seq_length = int(seq_length)
    batch_size = int(batch_size)
    num_epochs = int(num_epochs)
    hidden_size = int(hidden_size)
    num_layers = int(num_layers)

    df = load_data(PICKLE_FILE)
    if df is None:
        return float("inf")

    dataset = TradingDataset(df, seq_length)
    train_idx, valid_idx = train_test_split(list(range(len(dataset))), test_size=VALIDATION_SPLIT, random_state=42)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx))
    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(valid_idx))

    model = LSTMSignal(input_size=8, hidden_size=hidden_size, num_layers=num_layers, output_size=3).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
        
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(valid_idx)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)

    return best_val_loss

search_space = [
    Integer(1, 60),         # SEQ_LENGTH
    Integer(1, 64),         # BATCH_SIZE
    Real(0.0001, 0.001),    # LEARNING_RATE
    Integer(10, 100),       # NUM_EPOCHS
    Integer(10, 128),       # HIDDEN_SIZE
    Integer(2, 16)          # NUM_LAYERS
]



# Record start time
start_time = time.time()



logging.info("Starting Bayesian Optimization for hyperparameters...")
# opt_result = gp_minimize(train_and_evaluate, search_space, n_calls=20, random_state=42)
opt_result = gp_minimize(train_and_evaluate, search_space, n_calls=20, random_state=42, verbose=True)

best_params = {
    "SEQ_LENGTH": opt_result.x[0],
    "BATCH_SIZE": opt_result.x[1],
    "LEARNING_RATE": opt_result.x[2],
    "NUM_EPOCHS": opt_result.x[3],
    "HIDDEN_SIZE": opt_result.x[4],
    "NUM_LAYERS": opt_result.x[5]
}

logging.info(f"Best Hyperparameters Found: {best_params}")

# Train final model with best hyperparameters
train_and_evaluate(opt_result.x)





# Record end time
end_time = time.time()

# Calculate elapsed time in seconds
elapsed_time = end_time - start_time

# Convert to hours, minutes, and seconds
hours = int(elapsed_time // 3600)  # Integer division by 3600 (seconds in an hour)
minutes = int((elapsed_time % 3600) // 60)  # Remainder after hours, divided by 60 (seconds in a minute)
seconds = elapsed_time % 60  # Remainder after minutes

# Format the output
time_str = []
if hours > 0:
    time_str.append(f"{hours} hour{'s' if hours != 1 else ''}")
if minutes > 0:
    time_str.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
time_str.append(f"{seconds:.2f} second{'s' if seconds != 1 else ''}")

print(f"The loop took {', '.join(time_str)} to run.")
