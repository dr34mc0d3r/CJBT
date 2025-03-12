"""Torch LSTM to learn my personal patterns"""

# Grock discussion: https://x.com/i/grok?conversation=1898847484652265592



import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing

class TimeSeriesData:
    """
    Class to handle data preparation for time series classification.
    Purpose: Prepare and transform the dataset for LSTM training, including feature selection,
             data splitting, scaling, and sequence creation.
    """
    def __init__(self, df, window_size):
        """
        Initialize with dataframe and window size for sequence creation.
        
        Parameters:
        - df (pandas.DataFrame): Input dataframe with OHLCV, buy, sell, EMAs, and crossover points.
        - window_size (int): Number of past time steps to use for each sequence.
        """
        self.df = df
        self.window_size = window_size
        self.calculate_position()
        self.select_features()
        self.split_data()
        self.scale_data()
        self.create_sequences()


    def calculate_position(self):
        """
        Calculate position column based on signal to track holding status.
        Purpose: Help model understand valid actions (buy when not holding, sell when holding).
        """
        position = [0]
        for i in range(1, len(self.df)):
            if self.df['signal'][i-1] == 1 and position[-1] == 0:
                position.append(1)
            elif self.df['signal'][i-1] == 2 and position[-1] == 1:
                position.append(0)
            else:
                position.append(position[-1])
        self.df['position'] = position

    def select_features(self):
        """
        Select relevant features for model input.
        Purpose: Define which columns to use as input features, including OHLCV, EMAs, and position.
        """
        self.features = ['open', 'high', 'low', 'close', 'volume', '13EMA', '100EMA', 
                        '13EMACrossPoint', '100EMACrossPoint', 'position']
        self.target = 'signal'

    def split_data(self):
        """
        Split data into training and testing sets based on time.
        Purpose: Ensure temporal order is preserved, using 80% for training and 20% for testing.
        """
        train_size = int(len(self.df) * 0.8)
        self.train_df = self.df.iloc[:train_size]
        self.test_df = self.df.iloc[train_size:]

    def scale_data(self):
        """
        Scale features using MinMaxScaler to normalize input data.
        Purpose: Ensure all features are on the same scale for better model performance.
        """
        scaler = preprocessing.MinMaxScaler()
        self.train_features_scaled = scaler.fit_transform(self.train_df[self.features])
        self.test_features_scaled = scaler.transform(self.test_df[self.features])
        self.scaler = scaler

    def create_sequences(self):
        """
        Create sequences of data for training and testing.
        Purpose: Transform data into sequences of length window_size, with corresponding targets.
        """
        def create_sequence(data, df_target):
            X = []
            y = []
            for i in range(self.window_size, len(data)):
                seq = data[i - self.window_size:i, :]
                label = df_target[i]
                X.append(seq)
                y.append(label)
            return np.array(X), np.array(y)
        
        self.X_train, self.y_train = create_sequence(self.train_features_scaled, 
                                                   self.train_df[self.target])
        self.X_test, self.y_test = create_sequence(self.test_features_scaled, 
                                                 self.test_df[self.target])

class LSTMSignalPredictor(nn.Module):
    """
    Class to define and train the LSTM neural network for signal prediction.
    Purpose: Implement the LSTM model with methods for training and evaluation, including all hyperparameters for optimization.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, model_path, dropout_rate=0.2):
        """
        Initialize the LSTM model with hyperparameters.
        
        Parameters:
        - input_size (int): Number of features in each time step.
        - hidden_size (int): Number of units in LSTM hidden layer, affects model capacity (default: 100).
        - num_layers (int): Number of LSTM layers, controls depth (default: 1).
        - output_size (int): Number of output classes (3 for buy, sell, hold).
        - dropout_rate (float): Dropout rate for regularization, prevents overfitting (default: 0.2).
        """
        super(LSTMSignalPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                          num_layers=num_layers, batch_first=True)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.model_path = model_path

    def forward(self, x):
        """
        Forward pass through the network.
        Purpose: Process input sequence through LSTM and fully connected layer to get predictions.
        
        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).
        
        Returns:
        - torch.Tensor: Output predictions of shape (batch_size, output_size).
        """
        output, _ = self.lstm(x)
        last_output = output[:, -1, :]
        last_output = self.dropout(last_output)
        prediction = self.fc(last_output)
        return prediction

    def train(self, X_train, y_train, batch_size, numEpochs, learning_rate, patience=10):
        """
        Train the model on training data.
        Purpose: Optimize model parameters using Adam optimizer and CrossEntropyLoss.
        
        Parameters:
        - X_train (numpy.ndarray): Training sequences.
        - y_train (numpy.ndarray): Training labels.
        - batch_size (int): Number of samples per batch, affects training speed and memory (default: 128).
        - numEpochs (int): Number of training epochs, controls training duration (default: 50).
        - learning_rate (float): Step size for optimizer, affects convergence (default: 0.001).
        - patience (int): Number of epochs with no improvement after which training stops (default: 10).
        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        X_train = torch.from_numpy(X_train).float().to(self.device)
        y_train = torch.from_numpy(y_train).long().to(self.device)
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        best_loss = float('inf')
        epochs_no_improve = 0
        
        for epoch in range(numEpochs):
            self.train()
            epoch_loss = 0
            for inputs, labels in train_dataloader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss = epoch_loss / len(train_dataloader)
            
            print(f'Epoch {epoch+1}/{numEpochs}, Loss: {epoch_loss:.4f}')
            
            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                epochs_no_improve = 0

                print(f"Saving best_loss: {best_loss:.10f}")
                self.save_model()

            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f'Early stopping triggered after epoch {epoch+1}')
                    break

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        Purpose: Compute accuracy on test set to assess generalization.
        
        Parameters:
        - X_test (numpy.ndarray): Test sequences.
        - y_test (numpy.ndarray): Test labels.
        """
        X_test = torch.from_numpy(X_test).float().to(self.device)
        y_test = torch.from_numpy(y_test).long().to(self.device)
        self.eval()
        with torch.no_grad():
            outputs = self(X_test)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y_test).sum().item() / len(y_test)
        print(f'Accuracy: {accuracy * 100:.2f}%')

    def save_model(self):
        checkpoint = {
            'lstm_state_dict': self.lstm.state_dict(),
            'future_head_state_dict': self.future_head.state_dict(),  # Save future head
            'scaler_data_min': torch.Tensor(self.scaler.data_min_),
            'scaler_data_max': torch.Tensor(self.scaler.data_max_),
            'scaler_feature_range': self.scaler.feature_range,
            'scaler_n_features_in_': self.scaler.n_features_in_
        }
        torch.save(checkpoint, self.model_path)
        print(f"Model saved to {self.model_path}")

    def clear_console(self):
        if os.name == 'nt':  # For Windows
            os.system('cls')
        else:  # For macOS and Linux
            os.system('clear')

    def print_hyperparameters(self):
        """
        Print all current hyperparameter values used in the model and training.
        """
        print("\n\n=== Current Hyperparameters ===")
        print(f"Input Size: {self.input_size}")
        print(f"Hidden Size: {self.hidden_size}")
        print(f"Number of Layers: {self.num_layers}")
        print(f"Dropout Rate: {self.dropout}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Model Path: {self.model_path}")
        print(f"Device: {self.device}")
        print("=== End of Hyperparameters ===\n")

    
    

def stocks_data(csv_path):
    df = pd.read_csv(csv_path)
    df['t'] = pd.to_datetime(df['t'])
    df.set_index('t', inplace=True)
    return df

def build_df_from_directory(root_dir, break_out_after=100000):
    # break_out_after how many csv files to load (100000 for all files)
    print("Training the model for forecasting...")
    files_with_ctime = []
    dataframes = []

    for dirname, _, filenames in os.walk(root_dir):
            for filename in filenames:
                file_path = os.path.join(dirname, filename)
                try:
                    ctime = os.path.getctime(file_path)
                    files_with_ctime.append((ctime, file_path))
                except FileNotFoundError:
                    print(f"Papa couldn't find the file: {file_path}")
                except OSError as e:
                    print(f"Papa encountered an error with the file: {file_path}. Error: {e}")

    files_with_ctime.sort()
    sorted_file_paths = [file_path for _, file_path in files_with_ctime]

    break_out_after_counter = 0
    for path in sorted_file_paths:
        try:
            if break_out_after_counter > break_out_after:
                break
            df = stocks_data(path)
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

    print(len(large_dataframe))
    print(large_dataframe.columns)
    print(large_dataframe.info())
    print(large_dataframe.tail())

    return large_dataframe



# Example usage
if __name__ == "__main__":

    # Record start time
    start_time = time.time()

    
    # Load your dataframe
    # df = pd.read_csv('your_data.csv')
    model_path="lstm_forecaster.pth"

    root_dir = '/Users/chrisjackson/Desktop/DEV/python/CNN1/src/lstm2/data/1m/TSLA'
    df = build_df_from_directory(root_dir, 100000)
    
    # Initialize TimeSeriesData
    window_size = 60  # Hyperparameter: Sequence length, affects model memory (try 20-100)
    data = TimeSeriesData(df, window_size)
    
    # Initialize LSTMSignalPredictor
    input_size = len(data.features)
    hidden_size = 100  # Hyperparameter: LSTM units, affects capacity (try 50-200)
    num_layers = 1  # Hyperparameter: Number of LSTM layers, controls depth (try 1-3)
    output_size = 3
    model = LSTMSignalPredictor(input_size, hidden_size, num_layers, output_size, model_path)

    # Clear the console at the beginning of the script
    model.clear_console()
    
    # Train the model
    batch_size = 128  # Hyperparameter: Batch size, affects training speed (try 64-256)
    numEpochs = 50  # Hyperparameter: Number of epochs, controls training duration (try 20-100)
    learning_rate = 0.001  # Hyperparameter: Learning rate, affects convergence (try 0.0001-0.01)

    
    # -------------------- Main Loop --------------------
    try:

        # Train if no model exists
        if not os.path.exists(model.model_path):

            model.train(data.X_train, data.y_train, batch_size, numEpochs, learning_rate)

            model.print_hyperparameters()  # Print hyperparameters after plots
    
        # Evaluate the model
        model.evaluate(data.X_test, data.y_test)


    except FileNotFoundError as e:
        print(e)

    # -------------------- Main Loop --------------------



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

