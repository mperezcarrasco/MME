import torch
import torch.nn as nn
import torch.nn.functional as F
from grad_reverse import grad_reverse


class Classifier(nn.Module):
    """ 
    Classifier network for the MME model. 
    Same model as in the original implementation.
    """
    def __init__(self, num_class, inc, temp):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(inc, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, eta=0.1):
        x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc(x) / self.temp
        return x_out