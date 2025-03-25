# https://chat.deepseek.com/a/chat/s/1faba187-7de4-4266-8e9b-2da0d50c185e


import os
from datetime import datetime
import re

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F  # Add this with other imports

from imblearn.over_sampling import SMOTE

class DataManager():
    def __init__(self):
        self.large_dataframe = pd.DataFrame()
        self.pickleFilePath = ""
        # self.short_ema_period = 0
        # self.long_ema_period = 0

    def stocks_data(self, csv_path):
        df = pd.read_csv(csv_path)
        df['t'] = pd.to_datetime(df['t'])
        df.set_index('t', inplace=True)
        return df
    
    def build_df_from_directory(self, root_dir, break_out_after=100000):

        # Regular expression to capture the date in YYYY-MM-DD format
        date_pattern = re.compile(r'\d{4}-\d{2}-\d{2}')

        # Sort files by creation time to process them in chronological order
        files_with_date = []
        dataframes = []
        

        # Walk through the directory and process each file
        for dirname, _, filenames in os.walk(root_dir):
            for filename in filenames:
                # Search for a date pattern in the filename
                match = date_pattern.search(filename)
                if match:
                    date_str = match.group()  # Extract the date string, e.g., "2020-12-31"
                    try:
                        # Convert the date string to a datetime object
                        file_date = datetime.strptime(date_str, '%Y-%m-%d')
                        file_path = os.path.join(dirname, filename)
                        files_with_date.append((file_date, file_path))
                    except ValueError:
                        print(f"Papa encountered an error parsing the date in file: {filename}")
                else:
                    print(f"Papa couldn't find a date in file: {filename}")

        # Sort the list of files by the extracted date
        files_with_date.sort()

        # Create a sorted list of file paths based on the date
        sorted_file_paths = [file_path for _, file_path in files_with_date]

        break_out_after_counter = 1
        for path in sorted_file_paths:
            try:
                if break_out_after_counter > break_out_after:
                    break

                # print(f"Loading CSV for {path}")
                df = self.stocks_data(path)
                dataframes.append(df)
                break_out_after_counter += 1
            except pd.errors.EmptyDataError:
                print(f"Papa noticed that {path} is empty. Skipping this one.")
            except FileNotFoundError:
                print(f"Papa couldn't find the file: {path}. Skipping.")
            except Exception as e:
                print(f"Papa encountered an error with {path}: {e}. Skipping.")

        if dataframes:
            large_dataframe = pd.concat(dataframes, axis=0, ignore_index=False)
            print("Papa successfully combined all the data into one DataFrame!")
            print("Sorting the data...")
            large_dataframe = large_dataframe.sort_index()
        else:
            large_dataframe = pd.DataFrame()
            print("Papa found no data to combine. The Resulting DataFrame is empty.")

        # print(len(large_dataframe))
        # print(large_dataframe.columns)
        # print(large_dataframe.info())
        # print(large_dataframe.tail())

        return large_dataframe
    
    
    
    def save_dataframe_as_pickle(self, df):
        """
        Save a pandas DataFrame to a pickle file.

        Parameters:
            df (pd.DataFrame): The DataFrame to save.
            file_path (str): The file path (including filename) where the pickle will be stored.
        """

        df.to_pickle(self.pickleFilePath)
        print(f"DataFrame saved to {self.pickleFilePath}")

    def load_dataframe_from_pickle(self, pickleFilePath):
        """
        Load a pandas DataFrame from a pickle file if it exists,
        with error trapping to handle any issues during loading.

        Parameters:
            file_path (str): Path to the pickle file.

        Returns:
            pd.DataFrame or None: The loaded DataFrame if successful, 
                                or None if the file doesn't exist or an error occurs.
        """

        if not os.path.exists(pickleFilePath):
            print(f"File '{pickleFilePath}' does not exist.")
            return None

        try:
            df = pd.read_pickle(pickleFilePath)
            print(f"DataFrame loaded successfully from '{pickleFilePath}'.")
            return df
        except Exception as e:
            print(f"Error loading pickle file '{pickleFilePath}': {e}")
            return None




# Technical Indicators Functions ==============================================
def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.ffill()

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line  # Return histogram

def calculate_volatility(series, window=20):
    # return series.rolling(window).std().fillna(method='ffill')
    return series.rolling(window).std().ffill().bfill()

# Data Preprocessing ==========================================================
class FinancialDataPreprocessor:
    """Enhanced data processing with technical indicators and improved targets"""
    
    def __init__(self, root_path, file_path, lookback=60, test_size=0.2, target_window=3):
        self.root_path = root_path
        self.file_path = file_path
        self.lookback = lookback
        self.test_size = test_size
        self.target_window = target_window
        self.scaler = StandardScaler()
        self.dataManager = DataManager()
        
    def load_and_clean(self):
        """Load data and add technical features"""
        # df = pd.read_pickle(self.file_path)

        df = self.dataManager.load_dataframe_from_pickle(self.file_path)

        if df is not None:
            print(f"Loading data from File '{self.file_path}'")

            print(df.info(), df.columns)
            # df.glimpse()

        else:
            print(f"File '{self.file_path}' does not exist. Loading from CSV files")
            df = self.dataManager.build_df_from_directory(self.root_path, 500)

            df = df.drop(columns=['VolumeWeighted'])

            # technicalIndicators.add_ema(df, dataManager.short_ema_period, "short")
            # technicalIndicators.add_ema(df, dataManager.long_ema_period, "long")

            # technicalIndicators.add_all_features(df)

            print(df.info(), df.columns)
    

        
        # Add technical indicators
        df['RSI'] = calculate_rsi(df['Close'])
        df['MACD'] = calculate_macd(df['Close'])
        df['Volatility'] = calculate_volatility(df['Close'])
        
        # Handle missing values
        df.ffill(inplace=True)
        df.dropna(inplace=True)
        
        # Enhanced target: 3 consecutive bullish movements
        df['Target'] = (df['Close'].shift(-3) > df['Close']).astype(int)
        # df['Target'] = ((df['Close'].shift(-1) > df['Close']) &
        #                 (df['Close'].shift(-2) > df['Close'].shift(-1)) &
        #                 (df['Close'].shift(-3) > df['Close'].shift(-2))).astype(int)
        df.dropna(subset=['Target'], inplace=True)
        
        return df

    def prepare_data(self):
        """Time-series aware data preparation"""
        df = self.load_and_clean()
        
        # Sequential split
        split_idx = int(len(df) * (1 - self.test_size))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx - self.lookback:]
        
        # Scale features
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                       'RSI', 'MACD', 'Volatility']
        self.scaler.fit(train_df[feature_cols])
        
        train_scaled = self.scaler.transform(train_df[feature_cols]).astype(np.float32)
        test_scaled = self.scaler.transform(test_df[feature_cols]).astype(np.float32)
        
        # Create sequences
        train_X, train_y = self.create_sliding_windows(train_scaled, train_df['Target'].values)
        test_X, test_y = self.create_sliding_windows(test_scaled, test_df['Target'].values)
        
        # Class weight calculation
        pos_weight = (len(train_y) - sum(train_y)) / sum(train_y)
        
        return (train_X, train_y), (test_X, test_y), pos_weight

    def create_sliding_windows(self, data, targets):
        X, y = [], []
        # Add gap between sequences
        for i in range(0, len(data) - self.lookback, self.lookback//2):
            X.append(data[i:i+self.lookback])
            y.append(targets[i+self.lookback-1])
        return np.array(X), np.array(y)

# Model Architecture ==========================================================
class AttentionLSTM(nn.Module):
    """LSTM with attention mechanism for temporal pattern focusing"""
    
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 2)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_size]
        
        # Attention mechanism
        attn_weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        
        context = self.dropout(context)
        return self.classifier(context)
    
class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series data"""
    
    def __init__(self, features, targets):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.long)
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Training Framework ==========================================================
class LSTMTrainer:
    """Enhanced trainer with class weights and gradient clipping"""
    
    def __init__(self, model, train_loader, val_loader, pos_weight, learning_rate=0.001):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.criterion = nn.CrossEntropyLoss(
        #     weight=torch.tensor([1.0, pos_weight], dtype=torch.float32).to(self.device)
        # )
        alpha = torch.tensor([1.0, pos_weight], dtype=torch.float32).to(self.device)
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, pos_weight], device=self.device)
        )
        # self.criterion = nn.CrossEntropyLoss(
        #     weight=torch.tensor([1.0, pos_weight*2], dtype=torch.float32)
        # )
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        self.model.to(self.device)
        
    def train_epoch(self):
        """Training with gradient clipping"""
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
        return total_loss / total, correct / total
    
    def validate(self):
        """Validation evaluation"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
        return total_loss / total, correct / total
    
    def final_metrics(self, test_loader):
        """Calculate final evaluation metrics"""
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.numpy())
                
        print(f"Accuracy: {accuracy_score(all_targets, all_preds):.4f}")
        print(f"Precision: {precision_score(all_targets, all_preds):.4f}")
        print(f"Recall: {recall_score(all_targets, all_preds):.4f}")
        print(f"F1 Score: {f1_score(all_targets, all_preds):.4f}")
        print("Confusion Matrix:")
        print(confusion_matrix(all_targets, all_preds))
    
    def train(self, epochs=20, patience=3):
        """Training with early stopping"""
        best_val_loss = float('inf')
        no_improve = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
            else:
                no_improve += 1
                
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print("-" * 50)
            
            if no_improve >= patience:
                print("Early stopping triggered")
                break

# Add focal loss to handle class imbalance
class FocalLoss(nn.Module):
    """Implements Focal Loss for class imbalance mitigation
    
    Args:
        alpha (Tensor): Class weighting factors [1.0, pos_weight]
        gamma (float): Focusing parameter (2-5 works best)
    """
    def __init__(self, alpha=None, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            loss = self.alpha[targets] * loss
            
        return loss.mean()

# Main Execution ==============================================================
def main():
    # Configuration
    root_dir = '/Users/chrisjackson/Desktop/DEV/python/data/1m/TSLA'
    FILE_PATH = '/Users/chrisjackson/Desktop/DEV/python/data/pickleFile.pkl'
    LOOKBACK = 60
    BATCH_SIZE = 256
    EPOCHS = 30
    TEST_SIZE = 0.15
    
    # Data preparation
    preprocessor = FinancialDataPreprocessor(
        root_dir, FILE_PATH, 
        lookback=LOOKBACK,
        test_size=TEST_SIZE,
        target_window=3
    )
    (train_X, train_y), (test_X, test_y), pos_weight = preprocessor.prepare_data()
    
    # Diagnostic checks
    print(f"Class distribution - Train: {np.bincount(train_y)}")
    print(f"Positive class weight: {pos_weight:.2f}")
    
    # Time-series validation split
    # tscv = TimeSeriesSplit(n_splits=3)
    tscv = TimeSeriesSplit(n_splits=5, gap=LOOKBACK*2)
    for train_idx, val_idx in tscv.split(train_X):
        X_train, X_val = train_X[train_idx], train_X[val_idx]
        y_train, y_val = train_y[train_idx], train_y[val_idx]
    
    # Datasets & Loaders
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(test_X, test_y)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model initialization
    model = AttentionLSTM(input_size=8, hidden_size=128)
    trainer = LSTMTrainer(
        model, 
        train_loader, 
        val_loader,
        pos_weight=pos_weight,
        learning_rate=0.0008
    )
    
    # Training and evaluation
    trainer.train(epochs=EPOCHS)
    print("\nFinal Test Metrics:")
    trainer.final_metrics(test_loader)

if __name__ == "__main__":
    main()



