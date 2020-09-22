import os
import json
import torch
import numpy as np
import itertools
import random
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class EarlyStopping:
    """Early stopping as the convergence criterion.

        Args:
            args (string): hyperparameters for the training.
            patience (int): the model will stop if it not do improve in a patience number of epochs.

        Returns:
            stop (bool): if the model must stop.
            if_best (bool): if the model performance is better than the previous models.
    """
    def __init__(self, patience, directory):
        self.best_metric = 0.0
        self.counter = 0
        self.patience = patience
        self.directory = directory

    def count(self, ftr_ext, clf, metric):
        is_best = bool(metric > self.best_metric)
        self.best_metric = max(metric, self.best_metric)
        if is_best:
            self.counter = 0
            torch.save({'feature_extractor': ftr_ext.state_dict(),
                        'classifier': clf.state_dict()},
                        '{}/weights/trained_parameters.pth'.format(self.directory))
        else:
            self.counter += 1
        if self.counter > self.patience:
            stop = True
        else:
            stop = False
        return stop, is_best

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.1)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.1)
        m.bias.data.fill_(0)

def seed_everything(seed=1234):
    """Set the seeds for the results to be reproducible.
    Author: Benjamin Minixhofer
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def save_metrics(metrics, root_dir, mode='val'):
    """save all the metrics."""
    mt_dir = os.path.join(root_dir, 'metrics_{}.json'.format(mode))
    with open(mt_dir, 'w') as mt:
        json.dump(metrics, mt)

#Same scheduler provided in the original implementation.
def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma=0.0001,
                     power=0.75, init_lr=0.001):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (1 + gamma * iter_num) ** (- power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        i += 1
    return optimizer
