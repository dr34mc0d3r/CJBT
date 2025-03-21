"""
LSTM Model for Trading Signal Prediction

This module implements an LSTM-based time series model to predict trading signals (Buy/Sell) 
from OHLCV and EMA data. Supports both training and inference modes.
"""

import os
import time
import logging
import pickle
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('trading_lstm.log'), logging.StreamHandler()]
)

class Config:
    """Hyperparameters and configuration settings"""
    SEQUENCE_LENGTH = 60
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    DROPOUT_RATE = 0.2
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    BATCH_SIZE = 64
    EARLY_STOPPING_PATIENCE = 10
    MODEL_SAVE_PATH = "/Users/chrisjackson/Desktop/DEV/python/data/model.pth"
    DATA_PATH = "/Users/chrisjackson/Desktop/DEV/python/data/pickleFile.pkl"
    FEATURE_COLS = ['Open', 'High', 'Low', 'Close', 'Volume', 'VolumeWeighted', 'EMA_13', 'EMA_100']
    TARGET_COL = 'Signal'
    USE_CLASS_WEIGHTS = True  # New flag to enable/disable class balancing

class LSTMModel(nn.Module):
    """LSTM Network for sequence classification"""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout_rate: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # Take last timestep output
        return self.fc(out)

class DataHandler:
    def __init__(self, config: Config, is_training: bool = True):
        self.config = config
        self.is_training = is_training
        self.scaler = StandardScaler()
        self.last_modified = None

    def load_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], 
                                 Optional[np.ndarray], Optional[np.ndarray]]:
        """Load and preprocess data, returns (X_train, y_train, X_val, y_val)"""
        try:
            df = pd.read_pickle(self.config.DATA_PATH)

            print(f"Buy signals: {df[df['Signal'] == 1].describe()}")  # Buy signals
            print(f"Sell signals: {df[df['Signal'] == 2].describe()}")  # Sell signals
            print(f"Close price: {df[df['Close'] > 0].describe()}")
            
            if self.is_training:
                df = df[df[self.config.TARGET_COL] != 0]
                if len(df) < self.config.SEQUENCE_LENGTH * 2:
                    raise ValueError("Insufficient training data")

            features = df[self.config.FEATURE_COLS].values
            labels = df[self.config.TARGET_COL].values - 1 if self.is_training else None
            
            X, y = self._create_sequences(features, labels)
            if X is None or len(X) == 0:
                return None, None, None, None
                
            # Split into train/validation (80/20)
            split_idx = int(len(X) * 0.8)
            return (
                X[:split_idx], 
                y[:split_idx] if y is not None else None,
                X[split_idx:], 
                y[split_idx:] if y is not None else None
            )
        except (FileNotFoundError, KeyError, ValueError) as e:
            logging.error(f"Data loading error: {str(e)}")
            return None, None, None, None

    def _create_sequences(self, features: np.ndarray, labels: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Create time series sequences"""
        if len(features) < self.config.SEQUENCE_LENGTH:
            return np.array([]), np.array([]) if labels is not None else None
        
        X, y = [], []
        for i in range(len(features) - self.config.SEQUENCE_LENGTH + 1):
            X.append(features[i:i+self.config.SEQUENCE_LENGTH])
            if labels is not None:
                y.append(labels[i+self.config.SEQUENCE_LENGTH-1])
        
        X = np.array(X)
        if self.is_training:
            X = self._scale_features(X)

        # print(f"\n\n\n")
        # print(X, np.array(y) if y else None)
        return X, np.array(y) if y else None

    def _scale_features(self, X: np.ndarray) -> np.ndarray:
        """Normalize features using StandardScaler"""
        original_shape = X.shape
        X_2d = X.reshape(-1, original_shape[2])
        self.scaler.fit(X_2d)
        return self.scaler.transform(X_2d).reshape(original_shape)

class TrainingPipeline:
    """Handles model training and validation"""
    def __init__(self, model: LSTMModel, config: Config, device: torch.device, scaler: StandardScaler):
        self.model = model
        self.config = config
        self.device = device
        self.scaler = scaler  # Store scaler reference
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

        if config.USE_CLASS_WEIGHTS:
            class_counts = np.bincount(y_train)
            epsilon = 1e-6
            class_weights = 1.0 / (class_counts + epsilon)
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            self.criterion = nn.CrossEntropyLoss()


    def _save_checkpoint(self):
        """Save model and scaler state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler_mean': self.scaler.mean_,
            'scaler_scale': self.scaler.scale_,
            'config': self.config.__dict__
        }, self.config.MODEL_SAVE_PATH)

    def _create_dataloader(self, X: np.ndarray, y: np.ndarray) -> DataLoader:
        """Create PyTorch DataLoader with optional class balancing"""
        dataset = TensorDataset(
            torch.FloatTensor(X),
            torch.LongTensor(y)
        )
        
        if self.config.USE_CLASS_WEIGHTS:
            # Calculate class weights with epsilon to avoid division by zero
            class_counts = np.bincount(y)
            epsilon = 1e-6  # Small value to prevent zero division
            weights = 1.0 / (class_counts + epsilon)
            samples_weights = torch.tensor(weights[y], dtype=torch.float32)
            
            sampler = WeightedRandomSampler(
                weights=samples_weights,
                num_samples=len(samples_weights),
                replacement=True
            )
            
            return DataLoader(
                dataset,
                batch_size=self.config.BATCH_SIZE,
                sampler=sampler,
                pin_memory=True  # Helps with GPU transfer
            )
        else:
            return DataLoader(
                dataset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=True,
                pin_memory=True
            )

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        """Training loop with early stopping"""
        best_loss = float('inf')
        patience_counter = 0
        
        train_loader = self._create_dataloader(X_train, y_train)
        val_loader = self._create_dataloader(X_val, y_val)

        for epoch in range(self.config.NUM_EPOCHS):
            self.model.train()
            train_loss, train_acc = self._run_epoch(train_loader, training=True)
            
            self.model.eval()
            with torch.no_grad():
                val_loss, val_acc = self._run_epoch(val_loader, training=False)
            
            logging.info(
                f"Epoch {epoch+1}: "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
            )

            if val_loss < best_loss:
                best_loss = val_loss
                self._save_checkpoint()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    logging.info("Early stopping triggered")
                    break

    def _run_epoch(self, loader: DataLoader, training: bool) -> Tuple[float, float]:
        """Process one epoch"""
        total_loss, correct, total = 0, 0, 0
        
        for X_batch, y_batch in loader:
            if training:
                self.optimizer.zero_grad()
            
            outputs = self.model(X_batch.to(self.device))
            loss = self.criterion(outputs, y_batch.to(self.device))
            
            if training:
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch.to(self.device)).sum().item()
            total += y_batch.size(0)
        
        return total_loss / len(loader), correct / total


class InferencePipeline:
    """Handles real-time prediction"""
    def __init__(self, config: Config, device: torch.device):
        self.config = config
        self.device = device
        self.model, self.scaler = self._load_model()
        self.data_handler = DataHandler(config, is_training=False)

    def _load_model(self) -> Tuple[LSTMModel, StandardScaler]:
        """Load model and scaler state with safe deserialization"""
        import torch.serialization as ts
        from numpy import ndarray
        from numpy.core.multiarray import _reconstruct

        try:
            # Allow both ndarray type and reconstruction function
            with ts.safe_globals([_reconstruct, ndarray]):
                checkpoint = torch.load(
                    self.config.MODEL_SAVE_PATH,
                    map_location=self.device,
                    weights_only=True
                )
        except Exception as e:
            logging.error(f"Model loading failed: {str(e)}")
            raise

        # Reconstruct scaler with proper numpy array handling
        scaler = StandardScaler()
        scaler.mean_ = checkpoint['scaler_mean']
        scaler.scale_ = checkpoint['scaler_scale']
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = len(self.config.FEATURE_COLS)

        # Reconstruct model architecture
        model = LSTMModel(
            input_size=len(self.config.FEATURE_COLS),
            hidden_size=checkpoint['config']['HIDDEN_SIZE'],
            num_layers=checkpoint['config']['NUM_LAYERS'],
            dropout_rate=checkpoint['config']['DROPOUT_RATE']
        ).to(self.device)
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, scaler

    def monitor(self):
        """Monitor data file for changes and predict"""
        while True:
            try:
                current_modified = os.path.getmtime(self.config.DATA_PATH)
                if current_modified != self.data_handler.last_modified:
                    X, _ = self.data_handler.load_data()
                    if len(X) > 0:
                        self._predict(X[-1:])
                    self.data_handler.last_modified = current_modified
                time.sleep(30)
            except Exception as e:
                logging.error(f"Monitoring error: {str(e)}")

    def _predict(self, X: np.ndarray):
        """Make prediction on latest sequence"""
        with torch.no_grad():
            tensor_X = torch.FloatTensor(X).unsqueeze(0).to(self.device)
            outputs = self.model(tensor_X)
            predicted = torch.argmax(outputs).item() + 1
            logging.info(f"Predicted Signal: {predicted}")

class Evaluator:
    """Model performance evaluation"""
    @staticmethod
    def evaluate(model: LSTMModel, X: np.ndarray, y: np.ndarray, device: torch.device):
        """Calculate evaluation metrics with class handling"""
        model.eval()
        loader = DataLoader(TensorDataset(torch.FloatTensor(X), torch.LongTensor(y)), batch_size=32)
        
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                outputs = model(X_batch.to(device))
                all_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
                all_labels.extend(y_batch.numpy())

        # Convert to numpy arrays
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        
        # Handle empty predictions
        if len(np.unique(y_pred)) < 2:
            logging.warning("Model predicted only one class!")
            
        # Calculate metrics with zero_division
        metrics = {
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, zero_division=0),
            "Confusion Matrix": confusion_matrix(y_true, y_pred)
        }
        
        logging.info(f"Precision: {metrics['Precision']:.4f}")
        logging.info(f"Recall: {metrics['Recall']:.4f}")
        logging.info(f"F1-Score: {metrics['F1-Score']:.4f}")
        logging.info("Confusion Matrix:\n" + str(metrics['Confusion Matrix']))

        # Class distribution analysis
        unique, counts = np.unique(y_true, return_counts=True)
        logging.info(f"Class distribution (0=Buy, 1=Sell): {dict(zip(unique, counts))}")

if __name__ == "__main__":
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if os.path.exists(config.MODEL_SAVE_PATH):
        InferencePipeline(config, device).monitor()
    else:
        data_handler = DataHandler(config, is_training=True)
        X_train, y_train, X_val, y_val = data_handler.load_data()
        
        if X_train is None or y_train is None or X_val is None or y_val is None:
            raise ValueError("Failed to load training data or insufficient data")
            
        model = LSTMModel(
            input_size=len(config.FEATURE_COLS),
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            dropout_rate=config.DROPOUT_RATE
        ).to(device)
        
        # Pass the data handler's scaler to training pipeline
        trainer = TrainingPipeline(model, config, device, data_handler.scaler)
        trainer.train(X_train, y_train, X_val, y_val)
        Evaluator.evaluate(model, X_val, y_val, device)