import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Fix for Windows OMP Error

import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.model import NIDSModel
from src.train_model import load_processed_data, get_dataloader
from src.attack import fgsm_attack

# Config
BATCH_SIZE = 64
EPOCHS = 15 
LEARNING_RATE = 0.001
ADV_EPSILON = 0.1 # We train against attacks with 0.1 strength
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_PATH = os.path.join(config.PROJECT_ROOT, 'results', 'model_adversarial.pth')

def train_robust_model():
    print(f"Starting Adversarial Training on {DEVICE}...")
    
    # 1. Load Data
    X_train, y_train, X_test, y_test = load_processed_data()
    input_dim = X_train.shape[1]
    
    train_loader = get_dataloader(X_train, y_train, BATCH_SIZE, shuffle=True)
    
    # 2. Initialize Model
    model = NIDSModel(input_dim).to(DEVICE)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    start_time = time.time()
    
    # 3. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # --- STEP A: GENERATE ADVERSARIAL EXAMPLES ---
            # We generate attacks on the CURRENT batch
            # We use the model's current state to generate the attack
            adv_inputs = fgsm_attack(model, inputs, labels, epsilon=ADV_EPSILON)
            
            # --- STEP B: COMBINE DATA ---
            # Train on both Clean and Adversarial data
            combined_inputs = torch.cat((inputs, adv_inputs), dim=0)
            combined_labels = torch.cat((labels, labels), dim=0)
            
            # --- STEP C: STANDARD TRAINING STEP ---
            optimizer.zero_grad()
            
            outputs = model(combined_inputs)
            loss = criterion(outputs, combined_labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(train_loader):.4f}")
        
    print(f"Adversarial Training Complete in {time.time() - start_time:.2f}s")
    
    # 4. Save the Robust Model
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Robust Model saved to {SAVE_PATH}")

if __name__ == "__main__":
    train_robust_model()