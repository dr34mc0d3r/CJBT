import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

class Config:
    """Configuration class for model and training hyperparameters"""
    # Data parameters
    window_size = 60                # Sequence length for LSTM
    train_test_split = 0.8          # Train-validation split ratio
    
    # Model architecture
    input_size = None               # Will be set based on data (number of features)
    hidden_size = 128               # LSTM hidden state size
    num_layers = 2                  # Number of LSTM layers
    dropout = 0.3                   # Dropout probability
    use_bidirectional = False       # Whether to use bidirectional LSTM
    
    # Training parameters
    batch_size = 64                 # Mini-batch size
    epochs = 50                     # Maximum training epochs
    learning_rate = 0.001           # Learning rate
    early_stopping_patience = 5     # Early stopping patience
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TimeSeriesDataset(Dataset):
    """Custom dataset for time series sequences"""
    def __init__(self, data, targets, window_size):
        """
        Args:
            data: Input features (numpy array)
            targets: Target values (numpy array)
            window_size: Length of input sequences
        """
        self.data = data
        self.targets = targets
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.window_size]
        y = self.targets[idx+self.window_size]
        return torch.FloatTensor(x), torch.FloatTensor([y])

class LSTMModel(nn.Module):
    """LSTM-based binary classification model"""
    def __init__(self, config):
        super(LSTMModel, self).__init__()
        self.config = config
        
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=config.use_bidirectional,
            dropout=config.dropout if config.num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.hidden_size * (2 if config.use_bidirectional else 1), 1)

    def forward(self, x):
        h0 = torch.zeros(self.config.num_layers * (2 if self.config.use_bidirectional else 1), 
                        x.size(0), self.config.hidden_size).to(self.config.device)
        c0 = torch.zeros_like(h0)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # Take last timestep output
        out = self.fc(out)
        return out.squeeze()

class Trainer:
    """Handles model training and evaluation"""
    def __init__(self, model, config, train_loader, val_loader, class_weights):
        self.model = model.to(config.device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([class_weights[1]], device=config.device)
        )
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0

    def train_epoch(self):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        
        for inputs, labels in self.train_loader:
            inputs = inputs.to(self.config.device)
            labels = labels.to(self.config.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels.squeeze())
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels.squeeze()).sum().item()
            total += labels.size(0)

        return total_loss/total, correct/total

    def validate(self):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.config.device)
                labels = labels.to(self.config.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.squeeze())
                
                total_loss += loss.item() * inputs.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels.squeeze()).sum().item()
                total += labels.size(0)

        return total_loss/total, correct/total

    def train(self):
        for epoch in range(self.config.epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            print(f"Epoch {epoch+1}/{self.config.epochs}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.config.early_stopping_patience:
                    print("Early stopping triggered")
                    break

    def final_metrics(self, test_loader):
        self.model.load_state_dict(torch.load('best_model.pth'))
        self.model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.config.device)
                outputs = self.model(inputs)
                preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.squeeze().cpu().numpy())

        print("\nFinal Metrics:")
        print(f"Precision: {precision_score(all_labels, all_preds):.4f}")
        print(f"Recall: {recall_score(all_labels, all_preds):.4f}")
        print(f"F1 Score: {f1_score(all_labels, all_preds):.4f}")
        print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
        print("Confusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))

def main():
    # Load and prepare data
    df = pd.read_pickle("/Users/chrisjackson/Desktop/DEV/python/data/pickleFile.pkl")
    df = df.drop(columns=["SellSignal"]).ffill().bfill()  # Handle NaNs
    
    # Split data
    split_idx = int(len(df) * Config.train_test_split)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # Normalize features
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_df.drop(columns=["BuySignal"]))
    test_features = scaler.transform(test_df.drop(columns=["BuySignal"]))
    
    # Set input size in config
    Config.input_size = train_features.shape[1]
    
    # Calculate class weights
    class_weights = compute_class_weight('balanced', 
                                        classes=np.unique(train_df["BuySignal"]), 
                                        y=train_df["BuySignal"])
    
    # Create datasets
    train_dataset = TimeSeriesDataset(train_features, train_df["BuySignal"].values, Config.window_size)
    test_dataset = TimeSeriesDataset(test_features, test_df["BuySignal"].values, Config.window_size)
    
    # Split train into train/val
    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=Config.batch_size, shuffle=False)
    val_loader = DataLoader(val_subset, batch_size=Config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)
    
    # Initialize model and trainer
    model = LSTMModel(Config)
    trainer = Trainer(model, Config, train_loader, val_loader, class_weights)
    
    # Train and evaluate
    trainer.train()
    trainer.final_metrics(test_loader)

if __name__ == "__main__":
    main()