import torch.nn as nn
import torch.nn.functional as F


class TicTacToeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 128)  
        self.fc2 = nn.Linear(128, 64)  
        self.fc3 = nn.Linear(64, 32) 
        self.fc4 = nn.Linear(32, 9)  
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.relu((self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  
        return x

