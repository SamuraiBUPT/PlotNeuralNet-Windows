import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuplot import TorchPlot as tp

# 定义卷积神经网络
class NaiveConvNet(nn.Module):
    def __init__(self, config):
        super(NaiveConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        if config['dropout']:
            self.dropout = nn.Dropout(config['dropout_rate'])
        if config['enable_bn']:
            self.bn1 = nn.BatchNorm2d(6)  # Batch Normalization after conv1
            self.bn2 = nn.BatchNorm2d(16) # Batch Normalization after conv2
            self.bn3 = nn.BatchNorm1d(120)  # Batch Normalization after fc1
            self.bn4 = nn.BatchNorm1d(84)  # Batch Normalization after fc2
        self.fc3 = nn.Linear(84, 10)
        self.config = config

    def forward(self, x):
        if self.config['enable_bn']:
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.bn3(self.fc1(x)))
            x = F.relu(self.bn4(self.fc2(x)))
            if self.config['dropout']:
                x = self.dropout(x)
            x = self.fc3(x)
            return x
        else:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            if self.config['dropout']:
                x = self.dropout(x)
            x = self.fc3(x)
            return x
          
if __name__ == '__main__':
    config = {'enable_bn': True, 'dropout': True, 'dropout_rate': 0.5}
    # 定义一个网络
    net = NaiveConvNet(config)
    
    config2 = {}
    
    # initialize the ploter
    ploter = tp(config2)
    
    # prepare the input tensor
    input = torch.randn(16, 3, 32, 32)
    
    # plot
    ploter.plot(net, input)
