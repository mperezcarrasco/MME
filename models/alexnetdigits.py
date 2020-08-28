import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


import torchvision.models as models

class AlexnetDigits(nn.Module):
    """ 
    AlexNet adaptation for the digits dataset. 
    """
    def __init__(self, args, conv_dim=64):
        super(AlexnetDigits, self).__init__()
        if args.dataset == 'svhn':
            in_channels = 3
        else:
            in_channels = 1

        self.cnn1 = Conv(in_channels, conv_dim, 5, 2, 2)
        self.cnn2 = Conv(conv_dim, conv_dim*2, 5, 2, 2)
        self.cnn3 = Conv(conv_dim*2, conv_dim*4, 5, 2, 2)
        self.cnn4 = Conv(conv_dim*4, conv_dim*8, 4, 1, 0)

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)


    def features(self, x):
        h = F.leaky_relu(self.cnn1(x), 0.05)
        h = F.leaky_relu(self.cnn2(h), 0.05)
        h = F.leaky_relu(self.cnn3(h), 0.05)
        h = F.leaky_relu(self.cnn4(h), 0.05)
        return h.view(-1, 512)

    def classifier(self, x):
        h = self.fc1(x)
        h = self.fc2(x)
        return self.fc3(h)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    
class Conv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, padding, bn=True):
        super(Conv, self).__init__()
        self.bn = bn
        self.conv2d = nn.Conv2d(in_channels=dim_in, out_channels= dim_out,
                                kernel_size=kernel_size, stride=stride,
                                padding=padding, bias=True)
        self.bn2d = nn.BatchNorm2d(num_features=dim_out)
    def forward(self, x):
        if self.bn:
            return self.bn2d(self.conv2d(x))
        else:
            return self.conv2d(x)