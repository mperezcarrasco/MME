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
        if args.source == 'svhn' or args.target == 'svhn':
            in_channels = 3
        else:
            in_channels = 1
        
        self.features = nn.Sequential(Conv(in_channels, conv_dim, 5, 2, 2),
                                      nn.LeakyReLU(0.05, inplace=True),
                                      Conv(conv_dim, conv_dim*2, 5, 2, 2),
                                      nn.LeakyReLU(0.05, inplace=True),
                                      Conv(conv_dim*2, conv_dim*4, 5, 2, 2),
                                      nn.LeakyReLU(0.05, inplace=True),
                                      Conv(conv_dim*4, conv_dim*8, 4, 1, 0),
                                      nn.LeakyReLU(0.05, inplace=True))

        self.classifier = nn.Sequential(nn.Dropout(),
                                        nn.Linear(512, 256),
                                        nn.LeakyReLU(0.05, inplace=True),
                                        nn.Dropout(),
                                        nn.Linear(256, 128),
                                        nn.LeakyReLU(0.05, inplace=True),
                                        nn.Linear(128, 64))

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