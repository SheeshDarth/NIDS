import torch
import torch.nn as nn

class NIDSModel(nn.Module):
    def __init__(self, input_dim):
        """
        A simple Multi-Layer Perceptron (MLP) for Network Intrusion Detection.
        
        Args:
            input_dim (int): Number of input features (columns in your dataset).
        """
        super(NIDSModel, self).__init__()
        
        # Deep Neural Network Architecture
        # Layer 1: Input -> 128 Neurons (ReLU activation)
        self.layer1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3) # Dropout prevents overfitting
        
        # Layer 2: 128 -> 64 Neurons
        self.layer2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        
        # Layer 3: 64 -> 1 Neuron (Binary Output)
        self.output_layer = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid() # Squashes output between 0 and 1
        
    def forward(self, x):
        """
        Forward pass of the data through the network.
        """
        x = self.relu1(self.layer1(x))
        x = self.dropout1(x)
        
        x = self.relu2(self.layer2(x))
        x = self.dropout2(x)
        
        x = self.sigmoid(self.output_layer(x))
        return x