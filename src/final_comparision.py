import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Windows Fix

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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPSILONS = [0, 0.05, 0.1, 0.2, 0.3]

def evaluate_model(model_path, model_name):
    print(f"Evaluating {model_name}...")
    
    # Load Data
    _, _, X_test, y_test = load_processed_data()
    test_loader = get_dataloader(X_test, y_test, batch_size=1, shuffle=False)
    
    # Load Model
    input_dim = X_test.shape[1]
    model = NIDSModel(input_dim).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    accuracies = []
    
    for eps in EPSILONS:
        correct = 0
        total = 0
        
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            # Optimization: Only attack data the model initially gets RIGHT
            initial_output = model(data)
            pred = (initial_output > 0.5).float()
            
            if pred.item() != target.item():
                continue 
                
            # Attack
            perturbed_data = fgsm_attack(model, data, target, epsilon=eps)
            
            # Re-evaluate
            final_output = model(perturbed_data)
            final_pred = (final_output > 0.5).float()
            
            if final_pred.item() == target.item():
                correct += 1
            total += 1
            
        acc = correct / total if total > 0 else 0
        accuracies.append(acc)
        print(f"  Epsilon {eps}: {acc*100:.2f}%")
        
    return accuracies

if __name__ == "__main__":
    # Paths
    baseline_path = os.path.join(config.PROJECT_ROOT, 'results', 'model.pth')
    robust_path = os.path.join(config.PROJECT_ROOT, 'results', 'model_adversarial.pth')
    
    # Run Evaluations
    acc_baseline = evaluate_model(baseline_path, "Baseline Model")
    acc_robust = evaluate_model(robust_path, "Adversarial Model")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(EPSILONS, acc_baseline, "r*-", label="Standard NIDS (Vulnerable)")
    plt.plot(EPSILONS, acc_robust, "g*-", label="Defended NIDS (Robust)")
    
    plt.title("Adversarial Robustness Comparison")
    plt.xlabel("Attack Strength (Epsilon)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    
    save_path = os.path.join(config.PROJECT_ROOT, 'results', 'figures', 'final_comparison.png')
    plt.savefig(save_path)
    print(f"\nFinal Comparison Graph saved to: {save_path}")