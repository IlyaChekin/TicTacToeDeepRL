import torch.nn as nn
import torch.nn.functional as F


class TicTacToeModel(nn.Module):
    """
    A simple feedforward neural network for learning Tic-Tac-Toe strategies.
    
    Architecture:
        - Input: 9 features (Tic-Tac-Toe board state)
        - Hidden layers: Fully connected layers with ReLU activation
        - Dropout: Applied after each hidden layer for regularization
        - Output: 9 Q-values (one for each board position)
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 128)  # Input layer
        self.fc2 = nn.Linear(128, 128)  # Hidden layer 1
        self.fc3 = nn.Linear(128, 128)  # Hidden layer 2
        self.fc4 = nn.Linear(128, 64)  # Hidden layer 3
        self.fc5 = nn.Linear(64, 32)  # Hidden layer 4
        self.fc6 = nn.Linear(32, 9)  # Output layer
        self.dropout = nn.Dropout(0.2)  # Regularization

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, 9).
        
        Returns:
            torch.Tensor: Output tensor with shape (batch_size, 9),
            representing Q-values for each board position.
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = F.relu(self.fc5(x))
        x = self.fc6(x)  # Linear output layer (Q-values for actions)
        return x
    