#!/usr/bin/env python3
"""Time Series LSTM for Trading Signal Prediction.

This module provides classes and functions to train an LSTM model on historical
OHLCV and EMA data with trading signals, and to predict signals on new data
appended to a pickle file in real-time.

Usage:
    python script.py --train  # To train the model
    python script.py          # To predict using an existing model

Grok: https://x.com/i/grok?conversation=1902371477048930615
pip install pandas numpy torch scikit-learn joblib matplotlib optuna torch-tb-profiler

Train:
python src/LSTM_indicator/LSTMSignalPredictions.py --train --minority_factor 2.0 --pickle_file /Users/chrisjackson/Desktop/DEV/python/data/pickleFile.pkl --model_file /Users/chrisjackson/Desktop/DEV/python/data/model.pth --scaler_file /Users/chrisjackson/Desktop/DEV/python/data/scaler.pkl

Prediction
python src/LSTM_indicator/LSTMSignalPredictions.py --pickle_file /Users/chrisjackson/Desktop/DEV/python/data/pickleFile.pkl --model_file /Users/chrisjackson/Desktop/DEV/python/data/model.pth --scaler_file /Users/chrisjackson/Desktop/DEV/python/data/scaler.pkl
"""

import argparse
import os
import time
from typing import Tuple, Optional, List, Dict
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import joblib

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataHandler:
    """Handles data loading, preprocessing, and sequence creation for LSTM.

    Attributes:
        pickle_file (str): Path to the pickle file containing the DataFrame.
        sequence_length (int): Number of time steps in each sequence.
        features (List[str]): List of feature column names.
    """

    def __init__(self, pickle_file: str, sequence_length: int = 60) -> None:
        """Initialize the DataHandler.

        Args:
            pickle_file: Path to the pickle file.
            sequence_length: Length of sequences for LSTM input.
        """
        self.pickle_file = pickle_file
        self.sequence_length = sequence_length
        self.features = [
            "Open", 
            "High", 
            "Low", 
            "Close", 
            "Volume", 
            "VolumeWeighted", 
            "EMA_13", 
            "EMA_100"
            ]

    def load_data(self) -> pd.DataFrame:
        """Load DataFrame from the pickle file and filter by time range.

        Returns:
            DataFrame with datetime index and required columns, filtered to times
            between 14:30:00 and 20:59:00.

        Raises:
            FileNotFoundError: If the pickle file does not exist.
            ValueError: If required columns are missing or no data remains after filtering.
        """
        try:
            df = pd.read_pickle(self.pickle_file)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Pickle file not found: {self.pickle_file}") from e

        # Ensure the index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index 't' must be a DatetimeIndex")

        # Extract time component from the datetime index
        df_time = df.index.time
        start_time = pd.Timestamp("14:30:00").time()  # 14:30:00
        end_time = pd.Timestamp("20:59:00").time()    # 20:59:00

        # Filter rows where time is between 14:30:00 and 20:59:00 (inclusive)
        mask = (df_time >= start_time) & (df_time <= end_time)
        filtered_df = df.loc[mask].copy()

        if filtered_df.empty:
            raise ValueError(
                f"No data remains after filtering time range {start_time} to {end_time}"
            )

        # Verify required columns
        expected_columns = self.features + ["Signal"]
        missing_cols = [col for col in self.features if col not in filtered_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        return filtered_df

    def create_sequences(
        self, data: pd.DataFrame, include_target: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Create sequences for LSTM input.

        Args:
            data: Input DataFrame.
            include_target: Whether to include the target (Signal) column.

        Returns:
            Tuple of (sequences, targets) if include_target is True, else (sequences, None).

        Raises:
            ValueError: If data length is less than sequence_length.
        """
        if len(data) < self.sequence_length:
            raise ValueError(
                f"Data length ({len(data)}) is less than sequence_length ({self.sequence_length})"
            )

        sequences = []
        targets = [] if include_target else None
        target_col = "Signal"

        for i in range(len(data) - self.sequence_length + 1):
            seq = data.iloc[i : i + self.sequence_length][self.features].values
            sequences.append(seq)
            if include_target:
                targets.append(data.iloc[i + self.sequence_length - 1][target_col])

        return (
            np.array(sequences),
            np.array(targets, dtype=int) if include_target else None,
        )

    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        """Prepare training data with scaling and NaN/Inf checks.

        Returns:
            Tuple of (sequences, targets, scaler).

        Raises:
            FileNotFoundError: From load_data.
            ValueError: From load_data, create_sequences, or if NaN/Inf detected.
        """
        df = self.load_data()
        # Explicitly create a copy to avoid view/copy issues
        train_data = df.dropna(subset=["Signal"] + self.features).copy()
        
        # Check for NaN or Inf in features
        if not np.isfinite(train_data[self.features]).all().all():
            raise ValueError("Input features contain NaN or Inf values after dropna")
        
        if len(train_data) < self.sequence_length:
            raise ValueError(
                f"Insufficient data after dropping NaN rows: {len(train_data)} < {self.sequence_length}"
            )
        
        scaler = MinMaxScaler()
        # Convert features to float64 to match scaler output
        train_data[self.features] = train_data[self.features].astype("float64")
        # Assign scaled values
        train_data.loc[:, self.features] = scaler.fit_transform(train_data[self.features])
        
        # Check scaled data
        if not np.isfinite(train_data[self.features]).all().all():
            raise ValueError("Scaled features contain NaN or Inf values")
        
        X, y = self.create_sequences(train_data, include_target=True)
        return X, y, scaler
    
    def get_prediction_data(self, scaler: MinMaxScaler) -> np.ndarray:
        """Prepare the latest sequence for prediction.

        Args:
            scaler: Fitted scaler to transform the data.

        Returns:
            Array of shape (1, sequence_length, num_features).

        Raises:
            FileNotFoundError: From load_data.
            ValueError: From create_sequences or if insufficient valid data.
        """
        df = self.load_data()
        latest_data = df.iloc[-self.sequence_length :].copy()
        if latest_data[self.features].isna().any().any():
            raise ValueError("Latest data contains NaN values in features")
        latest_data[self.features] = scaler.transform(latest_data[self.features])
        X, _ = self.create_sequences(latest_data, include_target=False)
        return X  # Shape: (1, sequence_length, num_features)

class LSTMModel(nn.Module):
    """LSTM model for time series signal prediction.

    Attributes:
        num_layers (int): Number of LSTM layers.
        hidden_size (int): Size of the hidden state.
        lstm (nn.LSTM): The LSTM layer.
        fc (nn.Linear): Fully connected output layer.
    """

    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, output_size: int
    ) -> None:
        """Initialize the LSTM model.

        Args:
            input_size: Number of input features.
            hidden_size: Number of hidden units in LSTM.
            num_layers: Number of LSTM layers.
            output_size: Number of output classes (3: Hold, Buy, Sell).
        """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            Output tensor of shape (batch_size, output_size).
        """
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])  # Use the last time step
        return out

class Trainer:
    """Handles training of the LSTM model.

    Attributes:
        model (LSTMModel): The LSTM model to train.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer for training.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
    """

    def __init__(
        self,
        model: LSTMModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 0.001,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialize the Trainer.

        Args:
            model: The LSTM model to train.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            learning_rate: Learning rate for the optimizer.
            class_weights: Optional tensor of class weights for weighted loss.
        """
        self.model = model.to(DEVICE)
        self.criterion = (
            nn.CrossEntropyLoss(weight=class_weights)
            if class_weights is not None
            else nn.CrossEntropyLoss()
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train_epoch(self) -> float:
        """Train the model for one epoch.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        for X_batch, y_batch in self.train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
        return total_loss / len(self.train_loader.dataset)

    # def evaluate(self) -> Tuple[float, float]:
    #     """Evaluate the model on the validation set.

    #     Returns:
    #         Tuple of (average validation loss, accuracy).
    #     """
    #     self.model.eval()
    #     total_loss = 0.0
    #     correct = 0
    #     total = 0
    #     with torch.no_grad():
    #         for X_batch, y_batch in self.val_loader:
    #             X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
    #             outputs = self.model(X_batch)
    #             loss = self.criterion(outputs, y_batch)
    #             total_loss += loss.item() * X_batch.size(0)
    #             preds = torch.argmax(outputs, dim=1)
    #             correct += (preds == y_batch).sum().item()
    #             total += y_batch.size(0)
    #     avg_loss = total_loss / len(self.val_loader.dataset)
    #     accuracy = correct / total
    #     return avg_loss, accuracy
    
    def evaluate(self) -> Tuple[float, float, dict]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item() * X_batch.size(0)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        avg_loss = total_loss / len(self.val_loader.dataset)
        accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=[0, 1, 2])
        cm = confusion_matrix(all_labels, all_preds)
        metrics = {
            "precision": dict(zip(["Hold", "Buy", "Sell"], precision)),
            "recall": dict(zip(["Hold", "Buy", "Sell"], recall)),
            "f1": dict(zip(["Hold", "Buy", "Sell"], f1)),
            "confusion_matrix": cm
        }
        return avg_loss, accuracy, metrics

    def train(
        self, num_epochs: int, verbose: bool = True
    ) -> Dict[str, List[float]]:
        """Train the model for multiple epochs.

        Args:
            num_epochs: Number of epochs to train.
            verbose: Whether to print training progress.

        Returns:
            Dictionary with lists of train losses, val losses, and accuracies.
        """
        metrics = {"train_loss": [], "val_loss": [], "val_accuracy": []}
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            # val_loss, val_accuracy = self.evaluate()
            val_loss, val_accuracy, val_metrics = self.evaluate()
            metrics["train_loss"].append(train_loss)
            metrics["val_loss"].append(val_loss)
            metrics["val_accuracy"].append(val_accuracy)
            if verbose:
                print(f"Epoch {epoch + 1}/{num_epochs}, "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"Val Accuracy: {val_accuracy:.4f}")
                print(f"Val Metrics: {val_metrics}")

                # print(
                #     f"Epoch {epoch + 1}/{num_epochs}, "
                #     f"Train Loss: {train_loss:.4f}, "
                #     f"Val Loss: {val_loss:.4f}, "
                #     f"Val Accuracy: {val_accuracy:.4f}"
                # )
        return metrics

class Predictor:
    """Handles real-time signal prediction.

    Attributes:
        data_handler (DataHandler): Instance to handle data.
        model (LSTMModel): Trained LSTM model.
        scaler (MinMaxScaler): Fitted scaler for data normalization.
    """

    def __init__(
        self, pickle_file: str, model_file: str, scaler_file: str, sequence_length: int
    ) -> None:
        """Initialize the Predictor.

        Args:
            pickle_file: Path to the pickle file.
            model_file: Path to the trained model file.
            scaler_file: Path to the scaler file.
            sequence_length: Length of sequences for prediction.

        Raises:
            FileNotFoundError: If model or scaler files are missing.
        """
        self.data_handler = DataHandler(pickle_file, sequence_length)
        self.features = self.data_handler.features
        try:
            self.scaler = joblib.load(scaler_file)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Scaler file not found: {scaler_file}") from e

        # Load model (hyperparameters must match training)
        input_size = len(self.features)
        hidden_size = 50  # Must match training
        num_layers = 2    # Must match training
        output_size = 3
        self.model = LSTMModel(input_size, hidden_size, num_layers, output_size)
        try:
            self.model.load_state_dict(
                torch.load(model_file, map_location=DEVICE)
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model file not found: {model_file}") from e
        self.model.to(DEVICE)
        self.model.eval()

    def predict_latest(self) -> int:
        """Predict the signal for the latest data.

        Returns:
            Predicted signal (0: Hold, 1: Buy, 2: Sell).

        Raises:
            FileNotFoundError: From DataHandler.
            ValueError: From DataHandler.
        """
        X_pred = self.data_handler.get_prediction_data(self.scaler)
        X_tensor = torch.tensor(X_pred, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            output = self.model(X_tensor)
            pred = torch.argmax(output, dim=1).item()
        return pred


    def monitor_and_predict(self, interval: float = 30.0) -> None:
        """Monitor the pickle file and predict signals periodically.

        Args:
            interval: Time in seconds between predictions.
        """
        print(f"Starting prediction monitoring every {interval} seconds...")
        last_size = -1
        while True:
            try:
                df = self.data_handler.load_data()
                current_size = len(df)
                if current_size != last_size:
                    signal = self.predict_latest()
                    print(f"Predicted Signal: {signal} at {pd.Timestamp.now()}")
                    last_size = current_size
                else:
                    print("No new data, skipping prediction")
            except (FileNotFoundError, ValueError) as e:
                print(f"Error during prediction: {e}")
            time.sleep(interval)

def train_and_save(
    pickle_file: str,
    model_file: str,
    scaler_file: str,
    sequence_length: int = 60,
    hidden_size: int = 50,
    num_layers: int = 2,
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    minority_factor: float = 1.0,  # New parameter to adjust minority class weights
) -> None:
    """Train the model and save it along with the scaler.

    Args:
        pickle_file: Path to the pickle file.
        model_file: Path to save the trained model.
        scaler_file: Path to save the scaler.
        sequence_length: Length of sequences.
        hidden_size: Number of hidden units in LSTM.
        num_layers: Number of LSTM layers.
        num_epochs: Number of training epochs.
        batch_size: Batch size for training.
        learning_rate: Learning rate for the optimizer.
        minority_factor: Factor to multiply minority class weights (default is 1.0, no adjustment).

    Raises:
        FileNotFoundError: If pickle file is missing.
        ValueError: If data is insufficient.
    """
    # Load and prepare data
    data_handler = DataHandler(pickle_file, sequence_length)
    X, y, scaler = data_handler.get_training_data()

    # Split into train and validation sets
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    # Compute class weights
    class_counts = np.bincount(y_train, minlength=3)
    total_samples = len(y_train)
    num_classes = 3
    weights = total_samples / (num_classes * class_counts)
    print(f"Class Counts: \n{class_counts}")



    # Adjust weights for minority classes
    if minority_factor > 1.0:
        majority_class = np.argmax(class_counts)  # Identify the majority class (likely "Hold")
        for cls in range(num_classes):
            if cls != majority_class:  # Increase weights for "Buy" and "Sell"
                weights[cls] *= minority_factor

    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    # class_weights = torch.tensor([0.0001, 0.5, 0.5]).to(DEVICE)  # Example weights for [Hold, Buy, Sell]
    # criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Create DataLoaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    input_size = len(data_handler.features)
    output_size = 3  # Hold, Buy, Sell
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)

    # Train the model
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        learning_rate,
        class_weights=class_weights,  # Pass the adjusted weights
    )
    trainer.train(num_epochs)

    # Save model and scaler
    torch.save(model.state_dict(), model_file)
    joblib.dump(scaler, scaler_file)
    print(f"Model saved to {model_file}, Scaler saved to {scaler_file}")

# def train_and_save(
#     pickle_file: str,
#     model_file: str,
#     scaler_file: str,
#     sequence_length: int = 60,
#     hidden_size: int = 50,
#     num_layers: int = 2,
#     num_epochs: int = 100,
#     batch_size: int = 32,
#     learning_rate: float = 0.001,
# ) -> None:
#     """Train the model and save it along with the scaler.

#     Args:
#         pickle_file: Path to the pickle file.
#         model_file: Path to save the trained model.
#         scaler_file: Path to save the scaler.
#         sequence_length: Length of sequences.
#         hidden_size: Number of hidden units in LSTM.
#         num_layers: Number of LSTM layers.
#         num_epochs: Number of training epochs.
#         batch_size: Batch size for training.
#         learning_rate: Learning rate for the optimizer.

#     Raises:
#         FileNotFoundError: If pickle file is missing.
#         ValueError: If data is insufficient.
#     """
#     data_handler = DataHandler(pickle_file, sequence_length)
#     X, y, scaler = data_handler.get_training_data()

#     # Split into train and validation
#     train_size = int(0.8 * len(X))
#     X_train, X_val = X[:train_size], X[train_size:]
#     y_train, y_val = y[:train_size], y[train_size:]

#     # Compute class weights
#     class_counts = np.bincount(y_train, minlength=3)
#     total_samples = len(y_train)
#     num_classes = 3
#     weights = total_samples / (num_classes * class_counts)
#     class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

#     # Create DataLoaders
#     train_dataset = TensorDataset(
#         torch.tensor(X_train, dtype=torch.float32),
#         torch.tensor(y_train, dtype=torch.long),
#     )
#     val_dataset = TensorDataset(
#         torch.tensor(X_val, dtype=torch.float32),
#         torch.tensor(y_val, dtype=torch.long),
#     )
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#     # Initialize model
#     input_size = len(data_handler.features)
#     output_size = 3  # Hold, Buy, Sell
#     model = LSTMModel(input_size, hidden_size, num_layers, output_size)

#     # Train
#     trainer = Trainer(
#         model,
#         train_loader,
#         val_loader,
#         learning_rate,
#         class_weights=class_weights,
#     )
#     trainer.train(num_epochs)

#     # Save model and scaler
#     torch.save(model.state_dict(), model_file)
#     joblib.dump(scaler, scaler_file)
#     print(f"Model saved to {model_file}, Scaler saved to {scaler_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train or predict trading signals with an LSTM model."
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model and save it",
    )
    parser.add_argument(
        "--pickle_file",
        type=str,
        default="pickleFile.pkl",
        help="Path to the pickle file",
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default="model.pth",
        help="Path to the model file",
    )
    parser.add_argument(
        "--scaler_file",
        type=str,
        default="scaler.pkl",
        help="Path to the scaler file",
    )
    parser.add_argument(
        "--minority_factor",
        type=float,
        default=1.0,
        help="Factor to multiply minority class weights (e.g., 2.0 or 5.0)",
    )
    args = parser.parse_args()

    # Temporary test code in if __name__ == "__main__":
    # data_handler = DataHandler("/Users/chrisjackson/Desktop/DEV/python/data/pickleFile.pkl")
    # df = data_handler.load_data()
    # print(df.index.min(), df.index.max())  # Should show times within 14:30:00-20:59:00

    # print(df["Signal"].value_counts())
    # exit(0)

    if args.train:
        try:
            train_and_save(
                args.pickle_file,
                args.model_file,
                args.scaler_file,
                sequence_length=20, # 60
                hidden_size=50,
                num_layers=2,
                num_epochs=100,
                batch_size=32,
                learning_rate=0.0001, # .001
                minority_factor=args.minority_factor,  # Pass the minority factor
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"Training failed: {e}")
            exit(1)
    elif os.path.exists(args.model_file):
        try:
            predictor = Predictor(
                args.pickle_file,
                args.model_file,
                args.scaler_file,
                sequence_length=60,
            )
            predictor.monitor_and_predict(interval=1.0)
        except (FileNotFoundError, ValueError) as e:
            print(f"Prediction failed: {e}")
            exit(1)
    else:
        print(
            f"Model file {args.model_file} not found. Run with --train to train the model first."
        )
        exit(1)
