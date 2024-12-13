# train.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import urllib.request
import os
import joblib  # For saving scalers

# -----------------------------
# 1. Data Acquisition and Preprocessing
# -----------------------------

def download_dataset(url, filename):
    """
    Downloads a dataset from the specified URL if it doesn't already exist.
    """
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename}.")
    else:
        print(f"{filename} already exists.")

def load_and_preprocess_data(filename, seq_length=12):
    """
    Loads the dataset, normalizes it, creates sequences, and splits into training and testing sets.
    """
    # Load dataset
    df = pd.read_csv(filename, parse_dates=['Month'], index_col='Month')

    # Plot the data
    plt.figure(figsize=(12,4))
    plt.plot(df, label=filename.split('/')[-1].replace('.csv', ''))
    plt.title(f'{filename.split("/")[-1].replace(".csv", "")} Data')
    plt.xlabel('Month')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df.values)

    # Create sequences
    X = []
    y = []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i+seq_length])
        y.append(scaled_data[i+seq_length])

    X = np.array(X)
    y = np.array(y)

    return X, y, scaler

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
# 3. Training Function
# -----------------------------

def train_model(model, air_loader, sun_loader, criterion, optimizer, device, num_epochs=100):
    """
    Trains the Multi-Task LSTM model on both Air Passengers and Sunspots datasets.
    """
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        # Iterate over both loaders simultaneously
        for (air_batch, air_labels), (sun_batch, sun_labels) in zip(air_loader, sun_loader):
            air_batch = air_batch.to(device)
            air_labels = air_labels.to(device)

            sun_batch = sun_batch.to(device)
            sun_labels = sun_labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass for Air Passengers
            outputs_air = model(air_batch, task='air')
            loss_air = criterion(outputs_air, air_labels)

            # Forward pass for Sunspots
            outputs_sun = model(sun_batch, task='sun')
            loss_sun = criterion(outputs_sun, sun_labels)

            # Combine losses
            loss = loss_air + loss_sun

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(air_loader)
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# -----------------------------
# 4. Main Execution
# -----------------------------

if __name__ == "__main__":
    # -----------------------------
    # 4.1. Data Acquisition
    # -----------------------------

    # URLs for datasets
    air_passengers_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
    sunspots_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv'

    # Filenames
    air_filename = 'airline-passengers.csv'
    sun_filename = 'monthly-sunspots.csv'

    # Download datasets if not present
    download_dataset(air_passengers_url, air_filename)
    download_dataset(sunspots_url, sun_filename)

    # -----------------------------
    # 4.2. Data Preprocessing
    # -----------------------------

    SEQ_LENGTH = 12  # Using past 12 months to predict the next month
    air_X, air_y, air_scaler = load_and_preprocess_data(air_filename, SEQ_LENGTH)
    sun_X, sun_y, sun_scaler = load_and_preprocess_data(sun_filename, SEQ_LENGTH)

    # -----------------------------
    # 4.3. Train-Test Split
    # -----------------------------

    def train_test_split(X, y, train_size=0.8):
        """
        Splits the data into training and testing sets.
        """
        train_size = int(len(X) * train_size)
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        return X_train, y_train, X_test, y_test

    # Split Air Passengers data
    air_X_train, air_y_train, air_X_test, air_y_test = train_test_split(air_X, air_y)

    # Split Sunspots data
    sun_X_train, sun_y_train, sun_X_test, sun_y_test = train_test_split(sun_X, sun_y)

    # -----------------------------
    # 4.4. Create Datasets and DataLoaders
    # -----------------------------

    # Create datasets
    air_train_dataset = TimeSeriesDataset(air_X_train, air_y_train)
    air_test_dataset = TimeSeriesDataset(air_X_test, air_y_test)

    sun_train_dataset = TimeSeriesDataset(sun_X_train, sun_y_train)
    sun_test_dataset = TimeSeriesDataset(sun_X_test, sun_y_test)

    # Create DataLoaders
    BATCH_SIZE = 64

    air_train_loader = DataLoader(dataset=air_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    air_test_loader = DataLoader(dataset=air_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    sun_train_loader = DataLoader(dataset=sun_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    sun_test_loader = DataLoader(dataset=sun_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # -----------------------------
    # 4.5. Model Initialization
    # -----------------------------

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Initialize the model
    model = MultiTaskLSTM().to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # -----------------------------
    # 4.6. Training the Model
    # -----------------------------

    NUM_EPOCHS = 100
    train_model(model, air_train_loader, sun_train_loader, criterion, optimizer, device, NUM_EPOCHS)

    # -----------------------------
    # 4.7. Saving the Model and Scalers
    # -----------------------------

    # Create a directory to save the model and scalers
    save_dir = 'saved_models'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the model state_dict
    model_path = os.path.join(save_dir, 'multi_task_lstm.pth')
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

    # Save the scalers
    air_scaler_path = os.path.join(save_dir, 'air_scaler.pkl')
    sun_scaler_path = os.path.join(save_dir, 'sun_scaler.pkl')
    joblib.dump(air_scaler, air_scaler_path)
    joblib.dump(sun_scaler, sun_scaler_path)
    print(f'Scalers saved to {air_scaler_path} and {sun_scaler_path}')
