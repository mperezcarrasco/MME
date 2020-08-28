import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models


class VGG16(nn.Module):
    """ 
    VGG16 net pretrained on imagenet for Office dataset. 
    Same model as in the original implementation.
    """
    def __init__(self, args):
        super(VGG16, self).__init__()

        model = models.vgg16(pretrained=args.pretrain)
        self.features = model.features
        self.classifier = model.classifier

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512 * 7 * 7)
        x = self.classifier(x)
        return x