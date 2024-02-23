import torch.nn as nn 
import torch.nn.functional as F
import torch
from torch.autograd import Variable

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) #input channels, output channels, size of conv kernel
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

class XrayCNN(nn.Module):
    # Reference for the architecture is from this paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8982897/
    
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3,64,3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64,64,3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64,128,3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128,128,3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # need to flatten before dropout
        self.dropout = nn.Dropout(p = 0.2)

        self.dense = nn.Sequential(
            nn.Linear(self._get_conv_output((3, 256,256,)), 512), # a hack to avoid manual calculation, alternative - LazyModule
            nn.ReLU(),
            nn.Linear(512,2),
            nn.Softmax(dim = 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1) #all except batch dimension
        x = self.dropout(x)
        x = self.dense(x)

        return x

    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self.conv_layers(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size
    

class XrayCNN_mini(nn.Module):
    # Reference for the architecture is from this paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8982897/
    
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3,16,5),  #input channels, output channels, size of conv kernel
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16,32,5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32,64,5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # nn.Conv2d(64,128,5),
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2)
        )

        # need to flatten before dropout
        # self.dropout = nn.Dropout(p = 0.2)

        self.dense = nn.Sequential(
            nn.Linear(self._get_conv_output((3, 256,256,)), 128), # a hack to avoid manual calculation, alternative - LazyModule
            nn.ReLU(),
            nn.Linear(128,2),
            nn.Softmax(dim = 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1) #all except batch dimension
        # x = self.dropout(x)
        x = self.dense(x)

        return x

    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self.conv_layers(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size