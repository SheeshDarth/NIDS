import torch
import torch.nn as nn


def fgsm_attack(model, data, target, epsilon):
    """
    Performs the Fast Gradient Sign Method (FGSM) attack.
    
    Args:
        model: The trained PyTorch NIDS model.
        data: The input tensor (batch of network flows).
        target: The true labels.
        epsilon: The perturbation magnitude (attack strength).
    
    Returns:
        perturbed_data: The adversarial examples.
    """
    # 1. Enable gradient tracking for the input data
    # (Usually we only track weights, here we need input gradients)
    data.requires_grad = True
    
    # 2. Forward pass
    output = model(data)
    
    # 3. Calculate loss
    # We want to MAXIMIZE this loss to confuse the model
    loss = nn.BCELoss()(output, target)
    
    # 4. Zero all existing gradients
    model.zero_grad()
    
    # 5. Backward pass (Calculate gradient of loss w.r.t input)
    loss.backward()
    
    # 6. Get the sign of the data gradient
    data_grad = data.grad.data
    sign_data_grad = data_grad.sign()
    
    # 7. Create the perturbed image
    # Formula: x_adv = x + epsilon * sign(gradient)
    perturbed_data = data + epsilon * sign_data_grad
    
    # 8. Clip valid range
    # Our data was normalized (MinMax), so it should stay between 0 and 1
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    
    return perturbed_data