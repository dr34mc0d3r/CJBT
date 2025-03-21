#!/usr/bin/env python3
"""
time_series_lstm.py

This module implements a time series LSTM for algorithmic day trading.
It trains on a pickle file containing OHLCV, EMA_13, EMA_100 and Signal data.
Signals are 1 for Buy and 2 for Sell (0 for Hold is ignored).

If a trained model file exists, the module monitors the pickle file every 30 seconds,
appends new 1-minute data, and makes predictions using the saved model.
The module is implemented in an object-oriented manner with detailed logging,
hyperparameter definitions, and evaluation metrics (accuracy, precision, recall,
F1-score, confusion matrix) to assist in tuning performance.

avoids overlapping sliding windows between training and validation
if there is enough data. Else allows window overlap.

Author: Your Friendly Python Pro & Algorithmic Day Trader

MODEL_PATH = "/Users/chrisjackson/Desktop/DEV/python/data/model.pth"
PICKLE_FILE = "/Users/chrisjackson/Desktop/DEV/python/data/pickleFile.pkl"
"""




import os
import time
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# Global Hyperparameters
MODEL_FILE: str = "/Users/chrisjackson/Desktop/DEV/python/data/model.pth"
PICKLE_FILE: str = "/Users/chrisjackson/Desktop/DEV/python/data/pickleFile.pkl"
WINDOW_SIZE: int = 195  # 60 number of time steps in each input sequence
BATCH_SIZE: int = 32
NUM_EPOCHS: int = 50
LEARNING_RATE: float = 0.0001
HIDDEN_SIZE: int = 64
NUM_LAYERS: int = 4 # 2
DROPOUT: float = 0.2
NUM_CLASSES: int = 2  # Buy and Sell

# Feature columns to be used (excluding the Signal column)
FEATURE_COLUMNS: List[str] = ["Open", "High", "Low", "Close", "Volume", "EMA_13", "EMA_100"]


def get_device() -> torch.device:
    """
    Checks for GPU availability and returns the appropriate device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TimeSeriesDataset(Dataset):
    """
    Dataset class for time series data using sliding window approach.
    Generates sequences of features and the corresponding label from a contiguous slice
    of the filtered data (ignoring rows where Signal==0).

    Attributes:
        sequences (List[np.ndarray]): List of input feature sequences.
        labels (List[int]): Corresponding labels.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        window_size: int,
        start_idx: int = 0,
        end_idx: Optional[int] = None
    ) -> None:
        """
        Initializes the dataset by creating sliding windows from a specified slice of the provided DataFrame.

        Args:
            data (pd.DataFrame): DataFrame containing OHLCV, EMA, and Signal columns.
            window_size (int): Number of time steps for each input sequence.
            start_idx (int): Starting index (in the filtered data) to consider.
            end_idx (Optional[int]): Ending index (in the filtered data) to consider.
                If None, uses the full length.
        """
        # Filter out rows with Signal==0 (only keep Buy (1) and Sell (2))
        filtered_data = data[data["Signal"] != 0].copy()
        filtered_data.reset_index(drop=True, inplace=True)

        if end_idx is None or end_idx > len(filtered_data):
            end_idx = len(filtered_data)

        self.sequences: List[np.ndarray] = []
        self.labels: List[int] = []

        # Check that the slice has at least window_size+1 rows.
        if end_idx - start_idx < window_size + 1:
            raise ValueError(f"Not enough data in the specified slice to form a sliding window. end_idx({end_idx}) - start_idx({start_idx}) < window_size({window_size}) + 1\n {end_idx - start_idx} < {window_size + 1}")

        # Create sliding windows from the specified slice.
        for i in range(start_idx + window_size, end_idx):
            sequence = filtered_data.iloc[i - window_size:i][FEATURE_COLUMNS].values.astype(np.float32)
            # Convert labels: 1 (Buy) -> 0, 2 (Sell) -> 1
            label = int(filtered_data.iloc[i]["Signal"]) - 1
            self.sequences.append(sequence)
            self.labels.append(label)

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.sequences)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the sequence and label at the given index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of (sequence, label).
        """
        sequence = torch.tensor(self.sequences[index])
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return sequence, label


class LSTMModel(nn.Module):
    """
    LSTM-based model for time series classification.

    Attributes:
        lstm (nn.LSTM): LSTM layer for sequence processing.
        fc (nn.Linear): Fully connected layer for output predictions.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        num_classes: int,
    ) -> None:
        """
        Initializes the LSTM model with the given hyperparameters.

        Args:
            input_size (int): Number of features per time step.
            hidden_size (int): Number of features in the hidden state.
            num_layers (int): Number of recurrent layers.
            dropout (float): Dropout rate.
            num_classes (int): Number of output classes.
        """
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the LSTM model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, window_size, input_size).

        Returns:
            torch.Tensor: Logits for each class.
        """
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        out = self.fc(last_time_step)
        return out


class Trainer:
    """
    Trainer class to handle model training, validation, and evaluation.

    Attributes:
        model (LSTMModel): The LSTM model to be trained.
        device (torch.device): The device to use (CPU or GPU).
        optimizer (optim.Optimizer): Optimizer for training.
        criterion (nn.Module): Loss function.
    """

    def __init__(self, model: LSTMModel, device: torch.device, learning_rate: float) -> None:
        """
        Initializes the Trainer.

        Args:
            model (LSTMModel): The model to be trained.
            device (torch.device): The computation device.
            learning_rate (float): Learning rate for the optimizer.
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, data_loader: DataLoader) -> Tuple[float, float]:
        """
        Trains the model for one epoch.

        Args:
            data_loader (DataLoader): DataLoader for training data.

        Returns:
            Tuple[float, float]: Training loss and training accuracy.
        """
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for sequences, labels in data_loader:
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(sequences)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * sequences.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(data_loader.dataset)
        epoch_accuracy = accuracy_score(all_labels, all_preds)
        return epoch_loss, epoch_accuracy

    def validate_epoch(self, data_loader: DataLoader) -> Tuple[float, float]:
        """
        Validates the model for one epoch.

        Args:
            data_loader (DataLoader): DataLoader for validation data.

        Returns:
            Tuple[float, float]: Validation loss and validation accuracy.
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for sequences, labels in data_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * sequences.size(0)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(data_loader.dataset)
        epoch_accuracy = accuracy_score(all_labels, all_preds)
        return epoch_loss, epoch_accuracy

    def evaluate(
        self, data_loader: DataLoader
    ) -> Tuple[float, float, float, float, np.ndarray]:
        """
        Evaluates the model on the provided data loader.

        Args:
            data_loader (DataLoader): DataLoader for evaluation data.

        Returns:
            Tuple[float, float, float, float, np.ndarray]: Precision, recall, F1-score,
            accuracy, and confusion matrix.
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for sequences, labels in data_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(sequences)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        precision = precision_score(all_labels, all_preds, average="binary", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="binary", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
        accuracy = accuracy_score(all_labels, all_preds)
        conf_matrix = confusion_matrix(all_labels, all_preds)
        return precision, recall, f1, accuracy, conf_matrix


class Monitor:
    """
    Monitor class to continuously check for new data in the pickle file and make predictions.

    Attributes:
        pickle_file (str): Path to the pickle file.
        model_file (str): Path to the trained model file.
        window_size (int): Sequence window size.
        device (torch.device): Device for computation.
    """

    def __init__(self, pickle_file: str, model_file: str, window_size: int, device: torch.device) -> None:
        """
        Initializes the Monitor.

        Args:
            pickle_file (str): Path to the pickle file.
            model_file (str): Path to the trained model file.
            window_size (int): Sequence window size.
            device (torch.device): Computation device.
        """
        self.pickle_file = pickle_file
        self.model_file = model_file
        self.window_size = window_size
        self.device = device
        self.model: Optional[LSTMModel] = None
        self.load_model()

    def load_model(self) -> None:
        """
        Loads the trained model from the model file.
        """
        try:
            checkpoint = torch.load(self.model_file, map_location=self.device)
            self.model = LSTMModel(
                input_size=len(FEATURE_COLUMNS),
                hidden_size=checkpoint["hidden_size"],
                num_layers=checkpoint["num_layers"],
                dropout=checkpoint["dropout"],
                num_classes=NUM_CLASSES,
            )
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully!")
        except (FileNotFoundError, RuntimeError) as e:
            raise FileNotFoundError(f"Failed to load model from {self.model_file}: {e}")

    def predict(self, data: pd.DataFrame) -> Optional[int]:
        """
        Makes a prediction based on the latest data in the DataFrame.

        Args:
            data (pd.DataFrame): The DataFrame containing new OHLCV and EMA data.

        Returns:
            Optional[int]: The predicted signal (0 for Buy, 1 for Sell) or None if insufficient data.
        """
        if len(data) < self.window_size:
            print("Not enough data to make a prediction.")
            return None

        recent_data = data.iloc[-self.window_size:][FEATURE_COLUMNS].values.astype(np.float32)
        sequence = torch.tensor(recent_data).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(sequence)
            prediction = torch.argmax(outputs, dim=1).item()
        return prediction

    def start_monitoring(self) -> None:
        """
        Starts monitoring the pickle file for new data and makes predictions every 30 seconds.
        """
        print("Starting to monitor new data...")
        while True:
            try:
                df = pd.read_pickle(self.pickle_file)

                print(f"\n\n")
                print(df, df.info(), df.columns)

                # Convert 'Datetime' column to datetime format if it's not already
                df['Datetime'] = pd.to_datetime(df['Datetime'])
                # Set 'Datetime' as the index
                df.set_index('Datetime', inplace=True)

                # Check for missing timestamps
                all_minutes = pd.date_range(start=df.index.min(), end=df.index.max(), freq='T')
                missing_minutes = all_minutes.difference(df.index)
                # Display missing timestamps
                print(f"\n\nmissing_minutes: {missing_minutes}\n\n")

                # if missing_minutes then
                    # # Reindex to include all minutes
                    # combined_df = combined_df.reindex(all_minutes)

                    # # Optionally, fill missing values (e.g., forward fill)
                    # combined_df.fillna(method='ffill', inplace=True)


                print(f"\n\n")
                print(df, df.info(), df.columns)
                exit(0)

                if not all(col in df.columns for col in FEATURE_COLUMNS):
                    print("Pickle file missing required feature columns.")
                    time.sleep(30)
                    continue

                prediction = self.predict(df)
                if prediction is not None:
                    signal = prediction + 1  # Convert back: 0->Buy (1), 1->Sell (2)
                    print(f"Predicted Signal: {signal} (1 for Buy, 2 for Sell)")
                time.sleep(30)
            except Exception as e:
                print(f"Error during monitoring: {e}")
                time.sleep(30)


def train_model() -> None:
    """
    Trains the LSTM model on the dataset and saves the trained model.
    This version creates non-overlapping training and validation sets.
    """
    try:
        df = pd.read_pickle(PICKLE_FILE)






        # print(f"\n\n")
        # print(df, df.info(), df.columns)

        # Convert 'Datetime' column to datetime format if it's not already
        df.index = pd.to_datetime(df.index)

        # Check for missing timestamps
        all_minutes = pd.date_range(start=df.index.min(), end=df.index.max(), freq='min')
        missing_minutes = all_minutes.difference(df.index)
        # Display missing timestamps
        # print(f"\n\n-------------------------------------missing_minutes: {missing_minutes}\n\n")

        if not missing_minutes.empty:
            # Reindex to include all minutes
            df = df.reindex(all_minutes)

            # Optionally, fill missing values using forward fill
            df.ffill(inplace=True)
        




    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        print(f"Error loading pickle file {PICKLE_FILE}: {e}")
        return

    # Filter data to include only rows with Signal 1 or 2
    filtered_df = df[df["Signal"] != 0].copy()
    filtered_df.reset_index(drop=True, inplace=True)
    total_rows = len(filtered_df)

    # Check if there's enough data to form both training and validation sets.
    # We need at least 2*(WINDOW_SIZE+1) rows.
    if total_rows < 2 * (WINDOW_SIZE + 1):
        print(f"Not enough data to form both training and validation sets. "
              f"At least {2 * (WINDOW_SIZE + 1)} rows are required, but got {total_rows}.")
        return

    train_end = int(0.8 * total_rows)
    intended_gap = WINDOW_SIZE

    # Adjust gap if needed.
    if train_end + intended_gap >= total_rows:
        print("Not enough data for a validation set after applying the intended gap. Using a gap of 0 instead.")
        gap = 0
    else:
        gap = intended_gap

    val_start = train_end + gap

    try:
        train_dataset = TimeSeriesDataset(filtered_df, WINDOW_SIZE, start_idx=0, end_idx=train_end)
        val_dataset = TimeSeriesDataset(filtered_df, WINDOW_SIZE, start_idx=val_start, end_idx=total_rows)
    except ValueError as ve:
        print(f"An error occurred while forming sliding windows: {ve}")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = get_device()
    print(f"Using device: {device}")

    model = LSTMModel(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        num_classes=NUM_CLASSES,
    )

    trainer = Trainer(model, device, LEARNING_RATE)
    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = trainer.train_epoch(train_loader)
        val_loss, val_acc = trainer.validate_epoch(val_loader)
        print(
            f"Epoch {epoch}/{NUM_EPOCHS} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "state_dict": model.state_dict(),
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
            }
            torch.save(checkpoint, MODEL_FILE)
            print("Model checkpoint saved!")

    precision, recall, f1, accuracy, conf_matrix = trainer.evaluate(val_loader)
    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall:    {recall:.4f}")
    print(f"Validation F1 Score:  {f1:.4f}")
    print(f"Validation Accuracy:  {accuracy:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")


if __name__ == "__main__":
    try:
        if os.path.exists(MODEL_FILE):
            print("Trained model file found. Starting monitoring mode...")
            monitor = Monitor(
                pickle_file=PICKLE_FILE,
                model_file=MODEL_FILE,
                window_size=WINDOW_SIZE,
                device=get_device(),
            )
            monitor.start_monitoring()
        else:
            print("No trained model found. Starting training mode...")
            train_model()
    except KeyboardInterrupt:
        print("Exiting gracefully. Happy trading!")
    except Exception as error:
        print(f"An unexpected error occurred: {error}")
