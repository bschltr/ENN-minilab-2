import torch
import torch.nn as nn


class TorchMLP(nn.Module):
    """
    MLP with a single hidden layer, implemented in PyTorch.

    Architecture: Re-create your MLP in PyTorch
        Input are 2 dimensions
          - Linear layer from input to hidden
          - Sigmoid activation
          - Linear layer towards output
          - Sigmoid activation (or might use softmax later-on)
          - Output (3)

    Notes:
    - The forward pass already returns sigmoid outputs (probabilities).
    - Loss function and optimizer are defined outside this class.
    - Backpropagation is handled automatically by PyTorch's autograd.
    """
    def __init__(self, hidden_dim=5):
        """
        Parameters:
            hidden_dim : int
                Number of neurons in the hidden layer.
        """
        super().__init__()

        # --------------------------------------------------
        # Layers
        # --------------------------------------------------
        # TODO: Create layers (as done in the exercises) and activation functions

    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------
    def forward(self, x):
        """
        Forward pass of the network.

        x: Tensor of shape (N, 2)

        returns:
            y_hat: Tensor of shape (N, 3)
                   Sigmoid outputs (class-wise probabilities)
        """
        # TODO: Create forward pass
        # = instantiate the computation from input to output through the layers 
        # Calculate activations and apply activation functions afterwards.
        # Return the output (after the last activation function)

        # Initially, this will fail! There are no parameters and 
        # there is no working forward pass.
        # Pytorch will therefore fail creating autonmatically the backward pass / autograd.
        return 0