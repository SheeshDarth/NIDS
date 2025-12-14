import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

# Add src to system path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.model import NIDSModel

# 1. Configuration
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = os.path.join(config.PROJECT_ROOT, 'results', 'model.pth')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_processed_data():
    """Load numpy arrays saved in Phase 1"""
    print("Loading data...")
    X_train = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'y_train.npy'))
    X_test = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'y_test.npy'))
    return X_train, y_train, X_test, y_test

def get_dataloader(X, y, batch_size, shuffle=False):
    """Convert Numpy arrays to PyTorch DataLoaders"""
    tensor_x = torch.Tensor(X) # Transform to torch tensor
    tensor_y = torch.Tensor(y).unsqueeze(1) # Add dimension for binary classification
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train():
    print(f"Using Device: {DEVICE}")
    
    # 1. Load Data
    X_train, y_train, X_test, y_test = load_processed_data()
    input_dim = X_train.shape[1] # Should be 115 based on your previous output
    
    train_loader = get_dataloader(X_train, y_train, BATCH_SIZE, shuffle=True)
    test_loader = get_dataloader(X_test, y_test, BATCH_SIZE, shuffle=False)
    
    # 2. Initialize Model
    model = NIDSModel(input_dim).to(DEVICE)
    
    # 3. Define Loss and Optimizer
    criterion = nn.BCELoss() # Binary Cross Entropy
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting Training...")
    start_time = time.time()
    
    # 4. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(train_loader):.4f}")
        
    print(f"Training Complete in {time.time() - start_time:.2f}s")
    
    # 5. Save the Model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
    # 6. Final Evaluation
    evaluate(model, test_loader)

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    print(f"Test Set Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    train()

