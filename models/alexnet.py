import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

class Alexnet(nn.Module):
    """ 
    AlexNet pretrained on imagenet for Office dataset. 
    Same model as in the original implementation.
    """
    def __init__(self, args, pretrain=True):
        super(Alexnet, self).__init__()
        model = models.alexnet(pretrained=pretrain)
        self.features = model.features
    
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i),
                                       model.classifier[i])

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.classifier(x)
        print(x.shape)
        return x
