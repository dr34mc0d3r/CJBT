import os
from datetime import datetime
import re

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader

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




class FinancialDataPreprocessor:
    """
    Handles data loading, cleaning, and preprocessing of financial time series data.
    
    Attributes:
        file_path (str): Path to pickle file
        lookback (int): Number of historical time steps to use for prediction
        test_size (float): Proportion of data to use for testing
        target_col (str): Name of target column (Close for price prediction)
    """
    
    def __init__(self, df, lookback=60, test_size=0.2):
        self.df = df
        self.lookback = lookback
        self.test_size = test_size
        self.scaler = StandardScaler()
        
    def load_and_clean(self):
        """Load data from pickle file and handle missing values"""
        # df = pd.read_pickle(self.file_path)
        
        # Handle NaNs: Forward fill then drop remaining
        self.df.ffill(inplace=True)
        self.df.dropna(inplace=True)
        
        # Create target: 1 if next close > current close (bullish), else 0
        self.df['Target'] = (self.df['Close'].shift(-1) > self.df['Close']).astype(int)
        self.df.dropna(subset=['Target'], inplace=True)  # Remove last row with NaN target
        
        return self.df

    def prepare_data(self):
        """Main preprocessing pipeline"""
        self.df = self.load_and_clean()
        
        # Split data sequentially (time series)
        split_idx = int(len(self.df) * (1 - self.test_size))
        train_df = self.df.iloc[:split_idx]
        test_df = self.df.iloc[split_idx - self.lookback:]  # Include lookback for test sequences
        
        # Scale features
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.scaler.fit(train_df[feature_cols])
        
        train_scaled = self.scaler.transform(train_df[feature_cols])
        test_scaled = self.scaler.transform(test_df[feature_cols])
        
        # Create sliding window datasets
        train_X, train_y = self.create_sliding_windows(train_scaled, train_df['Target'].values)
        test_X, test_y = self.create_sliding_windows(test_scaled, test_df['Target'].values)
        
        return (train_X, train_y), (test_X, test_y)

    def create_sliding_windows(self, data, targets):
        """Convert time series to supervised learning format"""
        X, y = [], []
        for i in range(len(data) - self.lookback):
            X.append(data[i:i+self.lookback])
            y.append(targets[i+self.lookback-1])  # Predict next step from lookback window
        return np.array(X), np.array(y)

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series data"""
    
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.LongTensor(targets)
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class LSTMModel(nn.Module):
    """
    LSTM-based time series classifier
    
    Args:
        input_size: Number of features per time step
        hidden_size: LSTM hidden state size
        num_layers: Number of LSTM layers
        dropout: Dropout probability
    """
    
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 2)  # 2 classes (bullish/bearish)
        
    def forward(self, x):
        # LSTM returns: output, (hidden, cell)
        out, _ = self.lstm(x)
        # Take last time step's output
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.classifier(out)

class LSTMTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self, model, train_loader, val_loader, learning_rate=0.001):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def train_epoch(self):
        """Single training epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
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
    
    def train(self, epochs=10):
        """Full training loop"""
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print("-" * 50)
            
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

def main():

    LOOKBACK = 60  # 60 minutes window
    BATCH_SIZE = 128
    EPOCHS = 15
    TEST_SIZE = 0.2

    dataManager = DataManager()
    root_dir = '/Users/chrisjackson/Desktop/DEV/python/data/1m/TSLA'
    dataManager.pickleFilePath = '/Users/chrisjackson/Desktop/DEV/python/data/pickleFile.pkl'

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset

    df = dataManager.load_dataframe_from_pickle(dataManager.pickleFilePath)

    if df is not None:
        print(f"Loading data from File '{dataManager.pickleFilePath}'")

        print(df.info(), df.columns)
        # df.glimpse()

    else:
        print(f"File '{dataManager.pickleFilePath}' does not exist. Loading from CSV files")
        df = dataManager.build_df_from_directory(root_dir, 500)

        df = df.drop(columns=['VolumeWeighted'])

        # technicalIndicators.add_ema(df, dataManager.short_ema_period, "short")
        # technicalIndicators.add_ema(df, dataManager.long_ema_period, "long")

        # technicalIndicators.add_all_features(df)

        print(df.info(), df.columns)


    
    # Data preparation
    preprocessor = FinancialDataPreprocessor(df, lookback=LOOKBACK, test_size=TEST_SIZE)
    (train_X, train_y), (test_X, test_y) = preprocessor.prepare_data()
    
    # Split train into train/val
    train_X, val_X, train_y, val_y = train_test_split(
        train_X, train_y, test_size=0.2, shuffle=False  # No shuffle for time series
    )
    
    # Create datasets and loaders
    train_dataset = TimeSeriesDataset(train_X, train_y)
    val_dataset = TimeSeriesDataset(val_X, val_y)
    test_dataset = TimeSeriesDataset(test_X, test_y)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model and trainer
    model = LSTMModel(input_size=5, hidden_size=128, num_layers=2)
    trainer = LSTMTrainer(model, train_loader, val_loader, learning_rate=0.0005)
    
    # Train and evaluate
    trainer.train(epochs=EPOCHS)
    print("\nFinal Test Metrics:")
    trainer.final_metrics(test_loader)

if __name__ == "__main__":
    main()




