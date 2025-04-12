from torch import optim, nn
import torch
import torch.nn.functional as F

class myCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 61 * 61, 500)
        self.fc2 = nn.Linear(500, 340)
        self.fc3 = nn.Linear(340,200)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = nn.Dropout(0.5)(F.relu(self.fc1(x)))
        x = nn.Dropout(0.5)(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
    
class fastCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(4, 4)
        self.fc1 = nn.Linear(63 * 63 * 6 , 200)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = torch.flatten(x, 1) 
        x = self.fc1(x)
        return x

class OneLayerNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256 * 256 * 3 , 200)


    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x