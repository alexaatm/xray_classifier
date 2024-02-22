import torch.nn as nn 
import torch.nn.functional as F
import torch

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 61 * 61, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        # print(f'input X {x.size()}')
        x = self.pool(F.relu(self.conv1(x)))
        # print(f'X1 {x.size()}')
        x = self.pool(F.relu(self.conv2(x)))
        # print(f'X2 {x.size()}')
        x = torch.flatten(x, start_dim=1) #all except batch dimension
        # print(f'X3 {x.size()}')
        x = F.relu(self.fc1(x))
        # print(f'X4 {x.size()}')
        x = F.relu(self.fc2(x))
        # print(f'X5 {x.size()}')
        x = self.fc3(x)
        # print(f'X6 {x.size()}')
        return x
    
cnn = SimpleCNN()