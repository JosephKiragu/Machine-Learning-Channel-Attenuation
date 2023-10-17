
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the dataset
df = pd.read_csv('RFLFSODataFull_sample.csv')

# Data Preprocessing
# ... (Similar to code executed in the chat for Random Forest models)

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
