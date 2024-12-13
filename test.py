# test.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib  # For loading scalers
import os

# -----------------------------
# 1. Data Preprocessing
# -----------------------------

def load_and_preprocess_data(filename, scaler, seq_length=12):
    """
    Loads the dataset, normalizes it using the provided scaler, creates sequences.
    """
    # Load dataset
    df = pd.read_csv(filename, parse_dates=['Month'], index_col='Month')

    # Normalize the data
    scaled_data = scaler.transform(df.values)

    # Create sequences
    X = []
    y = []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i+seq_length])
        y.append(scaled_data[i+seq_length])

    X = np.array(X)
    y = np.array(y)

    return X, y

class TimeSeriesDataset(Dataset):
    """
    Custom Dataset for time series data.
    """
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -----------------------------
# 2. Model Definition
# -----------------------------

class MultiTaskLSTM(nn.Module):
    """
    Multi-Task LSTM Model with shared LSTM layers and task-specific output layers.
    Must match the architecture used during training.
    """
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(MultiTaskLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Shared LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Task-specific fully connected layers
        self.fc_air = nn.Linear(hidden_size, 1)
        self.fc_sun = nn.Linear(hidden_size, 1)

    def forward(self, x, task='air'):
        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = out[:, -1, :]  # (batch_size, hidden_size)

        if task == 'air':
            out = self.fc_air(out)
        elif task == 'sun':
            out = self.fc_sun(out)
        return out

# -----------------------------
# 3. Evaluation Function
# -----------------------------

def evaluate_model(model, data_loader, scaler, task='air', plot=True):
    """
    Evaluates the model on the specified task and returns predictions, actuals, and MSE.
    Optionally plots the results.
    """
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for batch, labels in data_loader:
            batch = batch.to(device)
            labels = labels.to(device)
            outputs = model(batch, task=task)
            predictions.append(outputs.cpu().numpy())
            actuals.append(labels.cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    # Inverse transform to get original scale
    predictions = scaler.inverse_transform(predictions)
    actuals = scaler.inverse_transform(actuals)
    mse = mean_squared_error(actuals, predictions)

    if plot:
        plt.figure(figsize=(12,4))
        plt.plot(actuals, label='Actual')
        plt.plot(predictions, label='Predicted')
        plt.title(f'{task.capitalize()} Prediction')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

    return predictions, actuals, mse

# -----------------------------
# 4. Main Execution
# -----------------------------

if __name__ == "__main__":
    # -----------------------------
    # 4.1. Load Scalers and Model
    # -----------------------------

    save_dir = 'saved_models'
    model_path = os.path.join(save_dir, 'multi_task_lstm.pth')
    air_scaler_path = os.path.join(save_dir, 'air_scaler.pkl')
    sun_scaler_path = os.path.join(save_dir, 'sun_scaler.pkl')

    # Load scalers
    air_scaler = joblib.load(air_scaler_path)
    sun_scaler = joblib.load(sun_scaler_path)

    # -----------------------------
    # 4.2. Device Configuration
    # -----------------------------

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # -----------------------------
    # 4.3. Load the Model
    # -----------------------------

    # Initialize the model
    model = MultiTaskLSTM().to(device)

    # Load the trained model state_dict
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f'Model loaded from {model_path}')

    # -----------------------------
    # 4.4. Data Preparation
    # -----------------------------

    # Filenames
    air_filename = 'airline-passengers.csv'
    sun_filename = 'monthly-sunspots.csv'

    # Sequence length
    SEQ_LENGTH = 12

    # Preprocess test data
    def load_and_preprocess_test(filename, scaler, seq_length=12):
        """
        Loads and preprocesses the test data.
        """
        X, y = load_and_preprocess_data(filename, scaler, seq_length)
        return X, y

    # Load and preprocess test data
    air_X_test, air_y_test = load_and_preprocess_test(air_filename, air_scaler, SEQ_LENGTH)
    sun_X_test, sun_y_test = load_and_preprocess_test(sun_filename, sun_scaler, SEQ_LENGTH)

    # Create test datasets
    air_test_dataset = TimeSeriesDataset(air_X_test, air_y_test)
    sun_test_dataset = TimeSeriesDataset(sun_X_test, sun_y_test)

    # Create DataLoaders
    BATCH_SIZE = 64
    air_test_loader = DataLoader(dataset=air_test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    sun_test_loader = DataLoader(dataset=sun_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # -----------------------------
    # 4.5. Evaluate the Model
    # -----------------------------

    # Evaluate on Air Passengers
    pred_air, actual_air, mse_air = evaluate_model(model, air_test_loader, air_scaler, task='air')
    print(f'Air Passengers Test MSE: {mse_air:.4f}')

    # Evaluate on Sunspots
    pred_sun, actual_sun, mse_sun = evaluate_model(model, sun_test_loader, sun_scaler, task='sun')
    print(f'Sunspots Test MSE: {mse_sun:.4f}')

    # -----------------------------
    # 4.6. Optional: Save Predictions
    # -----------------------------

    # Save predictions to CSV
    def save_predictions(predictions, actuals, filename):
        """
        Saves the actual and predicted values to a CSV file.
        """
        df = pd.DataFrame({
            'Actual': actuals.flatten(),
            'Predicted': predictions.flatten()
        })
        df.to_csv(filename, index=False)
        print(f'Predictions saved to {filename}')

    save_predictions(pred_air, actual_air, os.path.join(save_dir, 'air_passengers_predictions.csv'))
    save_predictions(pred_sun, actual_sun, os.path.join(save_dir, 'sunspots_predictions.csv'))
