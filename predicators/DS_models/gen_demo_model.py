import torch
import torch.nn as nn
import numpy as np
import os

class DynamicalSystem(nn.Module):
    """Simple dynamical system that converges to the origin."""
    def __init__(self):
        super(DynamicalSystem, self).__init__()
        # Create a negative definite damping matrix
        # Using a diagonal matrix with negative values for simplicity
        self.D = nn.Parameter(torch.diag(torch.tensor([-10, -10, -10])), requires_grad=False)
    
    def forward(self, x):
        """
        Compute velocity that converges to origin: dx/dt = D(x - 0)
        
        Args:
            x: Current state (batch_size, 3)
            
        Returns:
            out: Velocity (batch_size, 3)
        """
        # Ensure x is the right shape
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # Add batch dimension if needed
            
        # Calculate D(x - 0) = Dx
        out = torch.matmul(x, self.D.T)
        
        return out

# Create the model
model = DynamicalSystem()

# Set model to evaluation mode
model.eval()

# Create a directory for the model if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save the model
torch.save(model.state_dict(), "models/model.pt")

print("Model saved to models/model.pt")

