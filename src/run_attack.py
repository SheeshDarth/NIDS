import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import sys
import matplotlib.pyplot as plt

# Import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.model import NIDSModel
from src.attack import fgsm_attack
from src.train_model import load_processed_data, get_dataloader

# Config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = os.path.join(config.PROJECT_ROOT, 'results', 'model_adversarial.pth')
EPSILONS = [0, 0.05, 0.1, 0.2, 0.3] # Try different attack strengths

def test_attack():
    print(f"Running Attack on {DEVICE}")
    
    # 1. Load Data
    _, _, X_test, y_test = load_processed_data()
    # We only attack the Test set
    test_loader = get_dataloader(X_test, y_test, batch_size=1, shuffle=False)
    
    # 2. Load Model
    input_dim = X_test.shape[1]
    model = NIDSModel(input_dim).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval() # Set to evaluation mode
    
    accuracies = []
    
    for eps in EPSILONS:
        correct = 0
        total = 0
        
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            # Optimization: Only attack data the model initially gets RIGHT.
            # (There is no point attacking data it already misclassifies)
            initial_output = model(data)
            pred = (initial_output > 0.5).float()
            
            if pred.item() != target.item():
                continue # Skip already wrong predictions
                
            # Run FGSM Attack
            perturbed_data = fgsm_attack(model, data, target, epsilon=eps)
            
            # Re-classify the perturbed data
            final_output = model(perturbed_data)
            final_pred = (final_output > 0.5).float()
            
            if final_pred.item() == target.item():
                correct += 1
            
            total += 1
        
        # Calculate Robust Accuracy (Accuracy on adversarial examples)
        if total > 0:
            acc = correct / total
        else:
            acc = 0
            
        print(f"Epsilon: {eps}\tAccuracy: {acc*100:.2f}%")
        accuracies.append(acc)
        
    return EPSILONS, accuracies

if __name__ == "__main__":
    eps, accs = test_attack()
    
    # Plotting results (Optional but recommended)
    plt.figure(figsize=(10,6))
    plt.plot(eps, accs, "*-")
    plt.title("Adversarial Attack Success (FGSM)")
    plt.xlabel("Epsilon (Perturbation Amount)")
    plt.ylabel("Model Accuracy")
    plt.grid()
    plt.savefig(os.path.join(config.PROJECT_ROOT, 'results', 'figures', 'attack_results.png'))
    print("Plot saved to results/figures/attack_results.png")
