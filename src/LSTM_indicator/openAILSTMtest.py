import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# -------------------
# 1. DataLoader Class (Modified for Pickle)
# -------------------
class DataLoader:
    """Handles loading a single pickle file."""
    def __init__(self, pickle_path):
        """Initialize with the path to the pickle file."""
        self.pickle_path = pickle_path

    def load_data(self):
        """Load the pickle file into a DataFrame."""
        if not os.path.exists(self.pickle_path):
            raise FileNotFoundError(f"Pickle file not found at {self.pickle_path}")
        df = pd.read_pickle(self.pickle_path)
        # Ensure 't' is datetime and set as index (adjust column name if different)
        if 't' in df.columns:
            df['t'] = pd.to_datetime(df['t'])
            df.set_index('t', inplace=True)
        return df

# ------------------------
# 2. TimeSeriesDataset Class
# ------------------------
class TimeSeriesDataset(Dataset):
    """Custom Dataset for time series data with sequences and binary targets."""
    def __init__(self, df, segment_size, scaler=None):
        self.df = df
        self.segment_size = segment_size
        self.features = ['open', 'high', 'low', 'close', 'volume']
        self.scaler = scaler if scaler else StandardScaler()
        self.scaler.fit(self.df[self.features].values)

    def __len__(self):
        """Return the number of possible sequences, ensuring non-negative."""
        length = len(self.df) - self.segment_size
        return max(0, length)

    def __getitem__(self, idx):
        """Return a sequence and its target."""
        if idx >= len(self):
            raise IndexError
        sequence = self.df.iloc[idx:idx + self.segment_size][self.features].values
        sequence_scaled = self.scaler.transform(sequence).astype(np.float32)
        close_next = self.df['close'].iloc[idx + self.segment_size]
        close_prev = self.df['close'].iloc[idx + self.segment_size - 1]
        target = 1 if close_next > close_prev else 0
        sequence_tensor = torch.from_numpy(sequence_scaled)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        return sequence_tensor, target_tensor

# -------------------
# 3. LSTMModel Class
# -------------------
class LSTMModel(nn.Module):
    """LSTM model for binary classification, outputting logits."""
    def __init__(self, input_size=5, hidden_size=128, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)  # Output logits for BCEWithLogitsLoss

    def forward(self, x):
        """Forward pass returning logits."""
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# ----------------------
# 4. LSTMClassifier Class
# ----------------------
class LSTMClassifier:
    """Manages LSTM training, evaluation, and prediction with class imbalance handling."""
    def __init__(self, hidden_size=128, num_layers=2, dropout=0.3, learning_rate=0.001, model_path='lstm_model.pth'):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LSTMModel(input_size=5, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(self.device)
        self.pos_weight = torch.tensor([10.0]).to(self.device)  # Default, adjusted later
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=2, factor=0.5)
        self.load_model()

    def train(self, dataset, batch_size=32, epochs=10, val_dataset=None):
        """Train the model with optional validation."""
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            for sequences, targets in dataloader:
                sequences = sequences.float().to(self.device)
                targets = targets.float().unsqueeze(1).to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(sequences)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            epoch_loss = running_loss / len(dataloader)
            epoch_accuracy = correct / total
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f}')

            if val_dataset:
                val_loss, val_acc = self.evaluate(val_dataset)
                print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step(epoch_loss)
        self.save_model()

    def evaluate(self, dataset):
        """Evaluate the model on a dataset."""
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for sequences, targets in dataloader:
                sequences = sequences.float().to(self.device)
                targets = targets.float().unsqueeze(1).to(self.device)
                outputs = self.model(sequences)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        return running_loss / len(dataloader), correct / total

    def predict(self, dataset):
        """Generate predictions for inspection."""
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        predictions = []
        targets = []
        with torch.no_grad():
            for sequences, tgt in dataloader:
                sequences = sequences.float().to(self.device)
                outputs = torch.sigmoid(self.model(sequences))
                preds = (outputs > 0.5).float().cpu().numpy()
                predictions.extend(preds.flatten())
                targets.extend(tgt.numpy())
        return np.array(predictions), np.array(targets)

    def save_model(self):
        """Save the model state."""
        torch.save(self.model.state_dict(), self.model_path)
        print(f'Model saved to {self.model_path}')

    def load_model(self):
        """Load an existing model if available."""
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            print('Loaded existing model')
        else:
            print('No existing model found, starting from scratch')

# -------------------
# Main Execution
# -------------------
if __name__ == '__main__':
    # Configuration
    pickle_path = '/Users/chrisjackson/Desktop/DEV/python/data/pickleFile.pkl'  # Replace with your pickle file path
    segment_size = 60
    epochs = 10

    # Initialize objects
    dataloader = DataLoader(pickle_path)
    df = dataloader.load_data()
    
    print(f"Loaded data from {pickle_path}, rows: {len(df)}")
    
    # Check class balance
    up_count = df['close'].diff().gt(0).sum()
    down_count = len(df) - up_count - 1
    print(f"Up: {up_count}, Down: {down_count}")
    if up_count > 0 and down_count > 0:
        pos_weight = down_count / up_count
        lstm_classifier = LSTMClassifier(hidden_size=128, num_layers=2, dropout=0.3, learning_rate=0.001)
        lstm_classifier.pos_weight = torch.tensor([pos_weight]).to(lstm_classifier.device)
        lstm_classifier.criterion = nn.BCEWithLogitsLoss(pos_weight=lstm_classifier.pos_weight)
        print(f"Set pos_weight to {pos_weight:.2f}")
    else:
        print("Warning: No positive or negative class detected, using default pos_weight")
        lstm_classifier = LSTMClassifier(hidden_size=128, num_layers=2, dropout=0.3, learning_rate=0.001)

    # Train/validation split (80% train, 20% val)
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    
    if len(train_df) <= segment_size or len(val_df) <= segment_size:
        print(f"Insufficient data for split (train: {len(train_df)}, val: {len(val_df)})")
    else:
        train_dataset = TimeSeriesDataset(train_df, segment_size)
        val_dataset = TimeSeriesDataset(val_df, segment_size)
        
        # Train with validation
        lstm_classifier.train(train_dataset, batch_size=32, epochs=epochs, val_dataset=val_dataset)
        
        # Final evaluation with metrics
        preds, targets = lstm_classifier.predict(val_dataset)
        accuracy = (preds == targets).mean()
        precision = precision_score(targets, preds, zero_division=0)
        recall = recall_score(targets, preds, zero_division=0)
        f1 = f1_score(targets, preds, zero_division=0)
        cm = confusion_matrix(targets, preds)
        
        print("\nFinal Metrics on Validation Data:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print("Confusion Matrix:")
        print(cm)