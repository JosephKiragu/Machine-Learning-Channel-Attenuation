
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# Load the data
df = pd.read_csv("RFLFSODataFull_sample.csv")

# Drop rows with missing target values
df.dropna(subset=["FSO_Att", "RFL_Att"], inplace=True)

# Drop rows with any missing values
df.dropna(inplace=True)

# Data Subsetting (10% of the data, sampled randomly)
df = df.sample(frac=0.1, random_state=42)

# Separate features and target
X_rf = df.drop(["FSO_Att", "RFL_Att"], axis=1)
y_rf = df[["FSO_Att", "RFL_Att"]]

# Normalize features
scaler = StandardScaler()
X_rf_scaled = scaler.fit_transform(X_rf)

# Split the data
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf_scaled, y_rf, test_size=0.2, random_state=42)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd



# Data Preprocessing
# Load the data
df = pd.read_csv("RFLFSODataFull_sample.csv")

# Drop rows with missing target values
df.dropna(subset=["FSO_Att", "RFL_Att"], inplace=True)

# Drop rows with any missing values
df.dropna(inplace=True)

# Data Subsetting (10% of the data, sampled randomly)
df = df.sample(frac=0.1, random_state=42)

# Separate features and target
X_rf = df.drop(["FSO_Att", "RFL_Att"], axis=1)
y_rf = df[["FSO_Att", "RFL_Att"]]

# Normalize features
scaler = StandardScaler()
X_rf_scaled = scaler.fit_transform(X_rf)

# Split the data
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf_scaled, y_rf, test_size=0.2, random_state=42)

# Subset the data
subset_data = df.sample(frac=0.1, random_state=42)

# Separate features and targets
X_nn = subset_data.drop(['FSO_Att', 'RFL_Att'], axis=1).values
y_nn = subset_data[['FSO_Att', 'RFL_Att']].values

# Normalize features
scaler_nn = StandardScaler()
X_nn_scaled = scaler_nn.fit_transform(X_nn)

# Split data into training and testing sets
X_nn_train, X_nn_test, y_nn_train, y_nn_test = train_test_split(X_nn_scaled, y_nn, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_nn_train_tensor = torch.FloatTensor(X_nn_train)
y_nn_train_tensor = torch.FloatTensor(y_nn_train)
X_nn_test_tensor = torch.FloatTensor(X_nn_test)
y_nn_test_tensor = torch.FloatTensor(y_nn_test)

# Create DataLoader for training data
train_dataset = TensorDataset(X_nn_train_tensor, y_nn_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Implement Feedforward Neural Network (FNN) model
class FNNModel(nn.Module):
    def __init__(self, input_dim):
        super(FNNModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 2)  # Two output neurons for 'FSO_Att' and 'RFL_Att'
    
    def forward(self, x):
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# Initialize model, loss function, and optimizer
fnn_model = FNNModel(X_nn_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(fnn_model.parameters(), lr=0.001)

# Train FNN model
num_epochs = 50
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Zero parameter gradients
        optimizer.zero_grad()
        # Forward pass
        output = fnn_model(data)
        loss = criterion(output, target)
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

# Evaluate the model
# ... (Code for evaluation)


# RMSE and R-Squared as performance metrics
rmse = mean_squared_error(y_test_rf, y_pred_rf, squared=False)
r2 = r2_score(y_test_rf, y_pred_rf)
