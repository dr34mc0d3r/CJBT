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

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Check device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {DEVICE}")

# Model parameters (adjust for hyperparameter tuning)

# SEQ_LENGTH = 20
# This parameter defines the number of time steps used to create each input sequence.
# In a time series context, each sequence will consist of 20 consecutive data points.
# A larger sequence length might capture more historical context but can also increase computational cost.
SEQ_LENGTH = 60

# BATCH_SIZE = 64
# This defines the number of samples processed before the model's internal parameters are updated.
# A larger batch size can speed up training by utilizing vectorized operations but may require more memory.
# Conversely, a smaller batch size might improve model generalization but can be slower.
BATCH_SIZE = 64

# LEARNING_RATE = 0.001
# The learning rate controls how much the model weights are updated during training.
# A higher learning rate can speed up training but might overshoot minima,
# whereas a lower learning rate might lead to a more precise convergence but takes longer to train.
LEARNING_RATE = 0.0001

# NUM_EPOCHS = 20
# This parameter specifies the number of complete passes through the entire training dataset.
# More epochs can help the model learn better from the data, but too many epochs can lead to overfitting.
NUM_EPOCHS = 20

# HIDDEN_SIZE = 128
# The hidden size determines the number of features in the hidden state of the LSTM.
# A larger hidden size can capture more complex patterns in the data but also increases the model complexity and risk of overfitting.
HIDDEN_SIZE = 128

# NUM_LAYERS = 2
# This specifies the number of stacked LSTM layers in the model.
# Using multiple layers can allow the model to capture higher-level temporal features,
# but additional layers increase computational complexity and might require more data to train effectively.
NUM_LAYERS = 8

MODEL_PATH = "/Users/chrisjackson/Desktop/DEV/python/data/model.pth"
PICKLE_FILE = "/Users/chrisjackson/Desktop/DEV/python/data/pickleFile.pkl"
FOCAL_GAMMA = 2.0       # Focal loss gamma hyperparameter.
VALIDATION_SPLIT = 0.2  # Fraction of dataset used for validation.


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    
    Args:
        gamma (float): Focusing parameter for modulating factor (1-p).
        alpha (torch.Tensor, optional): Weighting factor for each class.
        reduction (str): Reduction method - 'mean' or 'sum'.
    """
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # Expected shape: (num_classes,)
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the focal loss between `inputs` and the ground truth `targets`.
        
        Args:
            inputs (torch.Tensor): Predictions (logits) of shape (batch_size, num_classes).
            targets (torch.Tensor): Ground truth labels of shape (batch_size).
        
        Returns:
            torch.Tensor: The computed focal loss.
        """
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            # Gather alpha weights for each target label.
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class TradingDataset(Dataset):
    """Custom PyTorch Dataset for Time-Series Trading Data."""
    
    def __init__(self, df: pd.DataFrame):
        self.X, self.y = self._prepare_sequences(df)

    def _prepare_sequences(self, df: pd.DataFrame):
        sequences, labels = [], []
        # Features: Open, High, Low, Close, Volume, VolumeWeighted, EMA_13, EMA_100
        features = ["Open", "High", "Low", "Close", "Volume", "VolumeWeighted", "EMA_13", "EMA_100"]
        for i in range(len(df) - SEQ_LENGTH):
            sequences.append(df.iloc[i:i + SEQ_LENGTH][features].values)
            labels.append(df.iloc[i + SEQ_LENGTH]["Signal"])
        return torch.tensor(np.array(sequences), dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMSignal(nn.Module):
    """LSTM Model for Time Series Classification."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super(LSTMSignal, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


def load_data(pickle_file: str) -> Optional[pd.DataFrame]:
    """Load and preprocess trading data from a pickle file."""
    if not os.path.exists(pickle_file):
        logging.error(f"Pickle file {pickle_file} not found.")
        return None

    with open(pickle_file, "rb") as f:
        df = pickle.load(f)

    # Ensure proper datetime index
    df.index = pd.to_datetime(df.index)
    # Filter for trading hours and copy to avoid SettingWithCopyWarning
    df = df.between_time("14:30", "20:59").copy()
    # Replace infinite values and drop rows with NaNs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


def compute_class_weights(dataset: TradingDataset) -> torch.Tensor:
    """
    Compute class weights inversely proportional to class frequencies.
    These weights are used in the loss function to penalize misclassification of minority classes.
    """
    y_np = dataset.y.numpy()
    counts = np.bincount(y_np)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / np.sum(weights)
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    logging.info(f"Computed class weights: {weights_tensor}")
    return weights_tensor


def get_sampler(dataset: TradingDataset) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler to balance the training data.
    Uses a smoothing factor (beta) to reduce oversampling aggressiveness.
    """
    y_np = dataset.y.numpy()
    class_sample_counts = np.bincount(y_np)
    beta = 0.5  # Smoothing factor between 0 and 1.
    weights = 1.0 / (np.power(class_sample_counts, beta) + 1e-6)
    sample_weights = weights[y_np]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)
    return sampler


def train_model(train_loader: DataLoader, valid_loader: DataLoader, train_dataset: TradingDataset):
    """Train the LSTM model using Focal Loss, monitoring validation loss."""
    
    model = LSTMSignal(input_size=8, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=3).to(DEVICE)
    class_weights = compute_class_weights(train_dataset)
    criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=class_weights, reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        # Training loop
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()
            total += targets.size(0)
        
        train_loss = total_loss / total
        train_acc = correct / total * 100

        # Validation loop
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                val_correct += (outputs.argmax(1) == targets).sum().item()
                val_total += targets.size(0)
        
        val_loss /= val_total
        val_acc = val_correct / val_total * 100

        logging.info(f"Epoch {epoch}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                     f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save model only if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            logging.info(f"Validation loss improved; model saved to {MODEL_PATH}")

    return model


def evaluate_model(model, data_loader):
    """Evaluate the model and log precision, recall, F1-score, and confusion matrix."""
    
    model.eval()
    all_targets, all_predictions = [], []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    cm = confusion_matrix(all_targets, all_predictions)
    report = classification_report(
        all_targets,
        all_predictions,
        target_names=['Hold', 'Buy', 'Sell'],
        digits=4,
        zero_division=0
    )
    logging.info(f"\nConfusion Matrix:\n{cm}")
    logging.info(f"\nClassification Report:\n{report}")


def predict(model, df: pd.DataFrame):
    """Make real-time predictions on new data."""
    
    model.eval()
    features = ["Open", "High", "Low", "Close", "Volume", "VolumeWeighted", "EMA_13", "EMA_100"]
    if len(df) < SEQ_LENGTH:
        logging.warning("Not enough data for prediction.")
        return

    input_seq = torch.tensor(df.iloc[-SEQ_LENGTH:][features].values, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(input_seq)
        _, predicted_signal = torch.max(output, 1)
    logging.info(f"Predicted Signal: {predicted_signal.item()}")


if __name__ == "__main__":
    df = load_data(PICKLE_FILE)
    if df is not None:
        # Create full dataset from DataFrame
        full_dataset = TradingDataset(df)
        
        # Split indices for training and validation
        indices = list(range(len(full_dataset)))
        train_indices, valid_indices = train_test_split(indices, test_size=VALIDATION_SPLIT, random_state=42)
        
        # Create sampler for training
        sampler = get_sampler(full_dataset)
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)
        
        # DataLoaders for training and validation
        train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
        valid_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, sampler=valid_sampler)
        
        if os.path.exists(MODEL_PATH):
            logging.info("Loading trained model for real-time predictions.")
            model = LSTMSignal(input_size=8, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=3).to(DEVICE)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.eval()
            while True:
                df = load_data(PICKLE_FILE)
                if df is not None:
                    predict(model, df)
                time.sleep(30)
        else:
            logging.info("No trained model found. Starting training process.")
            model = train_model(train_loader, valid_loader, full_dataset)
            evaluate_model(model, valid_loader)
