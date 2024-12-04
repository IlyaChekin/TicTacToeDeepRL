import torch.nn as nn
import torch.nn.functional as F


class TicTacToeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)  # 3x3 grid with 1 channel (board)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)  # 16 filters after conv1
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)  # 32 filters after conv2

        # Define a fully connected layer
        self.fc1 = nn.Linear(64 * 3 * 3, 128)  # Flattened output of the conv layers
        self.fc2 = nn.Linear(128, 9)  # Output layer: 9 Q-values (one for each action)

    def forward(self, x):
        # Reshape the input (board) to match (batch_size, 1, 3, 3) since we are using Conv2d
        x = x.view(-1, 1, 3, 3)
        
        # Pass through the convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the output from convolutional layers
        x = x.view(x.size(0), -1)
        
        # Pass through the fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Output Q-values for each action
        
        return x