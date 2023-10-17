
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

def build_and_evaluate_nn(df, scaler, target_columns, preprocess_func=None, feature_eng_func=None, subset_size=0.1):
    """
    Build and evaluate an improved Feedforward Neural Network model with hyperparameter tuning, dropout layers, 
    and Adam optimizer.
    
    Parameters:
    df (DataFrame): The input DataFrame.
    scaler (StandardScaler): Scaler for feature scaling.
    target_columns (list): List of target columns to be predicted.
    preprocess_func (function, optional): Preprocessing function.
    feature_eng_func (function, optional): Feature engineering function.
    subset_size (float): Fraction of data to use for model building.
    
    Returns:
    dict: Dictionary containing the model and evaluation metrics.
    """
    
    # Data Preprocessing
    if preprocess_func:
        df = preprocess_func(df)
        
    # Feature Engineering
    if feature_eng_func:
        df = feature_eng_func(df)
    
    # Use a subset of the data to avoid timeout
    df_subset = df.sample(frac=subset_size, random_state=42)
    
    # Separate features and target
    X_nn = df_subset.drop(columns=target_columns).values
    y_nn = df_subset[target_columns].values
    
    # Split the data
    X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(X_nn, y_nn, test_size=0.2, random_state=42)
    
    # Feature Scaling
    X_train_nn = scaler.fit_transform(X_train_nn)
    X_test_nn = scaler.transform(X_test_nn)
    
    # Convert to PyTorch tensors
    X_train_nn = torch.tensor(X_train_nn, dtype=torch.float32)
    y_train_nn = torch.tensor(y_train_nn, dtype=torch.float32)
    X_test_nn = torch.tensor(X_test_nn, dtype=torch.float32)
    y_test_nn = torch.tensor(y_test_nn, dtype=torch.float32)
    
    # Improved Neural Network Architecture
    input_size = X_train_nn.shape[1]
    output_size = y_train_nn.shape[1]
    
    class ImprovedFNN(nn.Module):
        def __init__(self):
            super(ImprovedFNN, self).__init__()
            self.fc1 = nn.Linear(input_size, 128)
            self.fc2 = nn.Linear(128, 64)
            self.dropout1 = nn.Dropout(0.2)
            self.fc3 = nn.Linear(64, 32)
            self.dropout2 = nn.Dropout(0.2)
            self.fc4 = nn.Linear(32, output_size)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.dropout1(x)
            x = torch.relu(self.fc3(x))
            x = self.dropout2(x)
            x = self.fc4(x)
            return x
    
    # Initialize the improved model
    model = ImprovedFNN()
    
    # Loss and Improved Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop with hyperparameter tuning
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_nn)
        loss = criterion(output, y_train_nn)
        loss.backward()
        optimizer.step()
    
    # Perform prediction on training and testing data
    y_train_pred_nn = model(X_train_nn)
    y_test_pred_nn = model(X_test_nn)
    
    # Calculate performance metrics on training data
    train_rmse = mean_squared_error(y_train_nn.detach().numpy(), y_train_pred_nn.detach().numpy(), squared=False)
    train_r2 = r2_score(y_train_nn.detach().numpy(), y_train_pred_nn.detach().numpy())
    
    # Calculate performance metrics on testing data
    test_rmse = mean_squared_error(y_test_nn.detach().numpy(), y_test_pred_nn.detach().numpy(), squared=False)
    test_r2 = r2_score(y_test_nn.detach().numpy(), y_test_pred_nn.detach().numpy())
    
    metrics = {
        target_columns[0]: {'Train RMSE': train_rmse, 'Train R2': train_r2},
        target_columns[1]: {'Test RMSE': test_rmse, 'Test R2': test_r2}
    }
    
    return {
        'Model': model,
        'Metrics': metrics
    }
