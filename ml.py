import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy as dc
import os
import logging

logger = logging.getLogger(__name__)


class LSTM(nn.Module):
    """Simple LSTM model for time series prediction."""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


def prepare_dataframe_for_lstm(df, n_steps, target_column='count'):
    """
    Prepare time series dataframe for LSTM training.
    Creates lagged features from the target column.
    
    Args:
        df: DataFrame with time series data
        n_steps: Number of lookback steps
        target_column: Column name to predict
    
    Returns:
        DataFrame with lagged features
    """
    df = dc(df)
    
    # Ensure target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")
    
    logger.info(f"Preparing data with {n_steps} lookback steps")
    logger.info(f"Original shape: {df.shape}")
    logger.info(f"NaNs in {target_column}: {df[target_column].isna().sum()}")

    # Create lag columns
    for i in range(1, n_steps + 1):
        df[f"{target_column}(t-{i})"] = df[target_column].shift(i)

    # Keep only target and lagged columns
    cols_to_keep = [target_column] + [f"{target_column}(t-{i})" for i in range(1, n_steps + 1)]
    df = df[cols_to_keep]
    
    # Drop rows with NaN (from shifting)
    df.dropna(inplace=True)

    logger.info(f"Shape after preparation: {df.shape}")
    logger.info(f"NaNs after dropna: {df.isna().sum().sum()}")

    return df


def train_lstm_model(
    df,
    target_column='count',
    lookback=7,
    split_ratio=0.80,
    skip_ratio=0.10,
    hidden_size=32,
    num_layers=2,
    learning_rate=0.001,
    batch_size=16,
    epochs=50,
    model_path='models/lstm_model.pth'
):
    """
    Train LSTM model on time series data.
    
    Args:
        df: DataFrame with time series (must have target_column)
        target_column: Column to predict (default: 'count')
        lookback: Number of past time steps to use
        split_ratio: Train/test split ratio
        skip_ratio: Skip initial data (to avoid early noise)
        hidden_size: LSTM hidden layer size
        num_layers: Number of LSTM layers
        learning_rate: Learning rate for optimizer
        batch_size: Training batch size
        epochs: Number of training epochs
        model_path: Path to save trained model
    
    Returns:
        model: Trained PyTorch model
        predictions: Test set predictions
        scaler: Fitted MinMaxScaler
        test_indices: Indices for test predictions
    """
    logger.info("=== Starting LSTM Training ===")
    
    # Prepare data with lagged features
    shift_df = prepare_dataframe_for_lstm(df, lookback, target_column)
    
    # Scale to [-1, 1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    shift_df_np = scaler.fit_transform(shift_df.values)

    # Separate X (lagged features) and Y (target)
    X = shift_df_np[:, 1:]  # All columns except first (target)
    Y = shift_df_np[:, 0]   # First column (target)

    # Flip to match LSTM input order (oldest to newest)
    X = np.flip(X, axis=1)

    # Split data
    split_index = int(len(X) * split_ratio)
    skip_index = int(len(X) * skip_ratio)
    
    X_train_np = X[skip_index:split_index]
    X_test_np = X[split_index:]
    Y_train_np = Y[skip_index:split_index]
    Y_test_np = Y[split_index:]

    logger.info(f"Training samples: {len(X_train_np)}, Test samples: {len(X_test_np)}")

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train_np.reshape((-1, lookback, 1)).copy()).float()
    Y_train = torch.tensor(Y_train_np.reshape((-1, 1)).copy()).float()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize model
    model = LSTM(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=1
    ).to(device)
    
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create DataLoader
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    logger.info(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)
            
            optimizer.zero_grad()
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    logger.info(f"Training complete! Final Loss: {avg_loss:.4f}")

    # Save model
    os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'lookback': lookback,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'target_column': target_column
    }, model_path)
    logger.info(f"Model saved to {model_path}")

    # Predict on test set
    X_test = torch.tensor(X_test_np.reshape((-1, lookback, 1)).copy()).float().to(device)
    
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test)

    # Inverse transform predictions
    inversed_preds_dummy = np.zeros((len(test_predictions), len(shift_df.columns)))
    inversed_preds_dummy[:, 0] = test_predictions.cpu().numpy().flatten()
    inversed_preds = scaler.inverse_transform(inversed_preds_dummy)[:, 0]

    # Get test indices from original dataframe
    test_start_idx = split_index + lookback
    test_indices = df.index[test_start_idx:test_start_idx + len(inversed_preds)]

    return model, inversed_preds, scaler, test_indices


def load_lstm_model(model_path='models/lstm_model.pth'):
    """
    Load a saved LSTM model.
    
    Args:
        model_path: Path to saved model
    
    Returns:
        model: Loaded PyTorch model
        scaler: Fitted scaler
        config: Model configuration dict
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract config
    config = {
        'lookback': checkpoint['lookback'],
        'hidden_size': checkpoint['hidden_size'],
        'num_layers': checkpoint['num_layers'],
        'target_column': checkpoint['target_column']
    }
    
    # Rebuild model
    model = LSTM(
        input_size=1,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        output_size=1
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scaler = checkpoint['scaler']
    
    logger.info(f"Model loaded from {model_path}")
    
    return model, scaler, config


def predict_future(model, scaler, last_sequence, n_steps, device=None):
    """
    Predict future values using autoregressive approach.
    
    Args:
        model: Trained LSTM model
        scaler: Fitted MinMaxScaler
        last_sequence: Last N values (unscaled) as numpy array
        n_steps: Number of steps to predict into future
        device: PyTorch device
    
    Returns:
        Array of predictions (unscaled)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    
    # Scale the last sequence
    last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1))
    
    # Start with the last sequence
    current_sequence = last_sequence_scaled.copy()
    predictions = []
    
    with torch.no_grad():
        for _ in range(n_steps):
            # Prepare input (flip for LSTM)
            input_seq = np.flip(current_sequence, axis=0)
            input_tensor = torch.FloatTensor(input_seq.reshape(1, -1, 1)).to(device)
            
            # Predict next value
            pred_scaled = model(input_tensor).cpu().numpy()[0, 0]
            predictions.append(pred_scaled)
            
            # Update sequence: remove oldest, add newest prediction
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = pred_scaled
    
    # Inverse transform predictions
    predictions = np.array(predictions).reshape(-1, 1)
    predictions_unscaled = scaler.inverse_transform(predictions).flatten()
    
    return predictions_unscaled


def evaluate_model(model, scaler, X_test, Y_test, lookback, device=None):
    """
    Evaluate model performance on test data.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        X_test: Test features (unscaled)
        Y_test: Test targets (unscaled)
        lookback: Lookback window size
        device: PyTorch device
    
    Returns:
        Dictionary of metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Scale and prepare data
    X_test_scaled = scaler.transform(X_test)
    Y_test_scaled = scaler.transform(Y_test.reshape(-1, 1))
    
    X_test_flipped = np.flip(X_test_scaled, axis=1)
    X_test_tensor = torch.FloatTensor(X_test_flipped.reshape(-1, lookback, 1)).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        predictions_scaled = model(X_test_tensor).cpu().numpy()
    
    # Inverse transform
    predictions = scaler.inverse_transform(predictions_scaled)
    
    # Calculate metrics
    mse = np.mean((Y_test - predictions.flatten()) ** 2)
    mae = np.mean(np.abs(Y_test - predictions.flatten()))
    rmse = np.sqrt(mse)
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'predictions': predictions.flatten(),
        'actual': Y_test
    }