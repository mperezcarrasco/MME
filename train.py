import torch
from torch.optim import SGD
from torch.nn import functional as F
#from torch.utils.tensorboard import SummaryWriter

from models.main import build_network, build_classifier
from sklearn.metrics import accuracy_score
from utils.utils import EarlyStopping, weights_init, save_metrics, inv_lr_scheduler
from itertools import cycle
import numpy as np


class Trainer(object):
    def __init__(self, args, 
                       device, 
                       writer, 
                       directory,
                       dl_sup_s,
                       dl_sup_t, 
                       dl_unsup_t, 
                       dl_val_t):
        """Trainer class for "Semi-supervised Domain Adaptation via Minimax Entropy",
        Saito K., Kim D., Sclaroff S., Darrel T., Saenko K., ICCV 2019.
        ref: https://arxiv.org/abs/1904.06487.
    
        Args:
            args (string): hyperparameters for the training.
            device (string): 'cuda' if available. Else 'cpu'.
            directory (str): directory to storage the results of the experiment.
            dl_sup_s (data.DataLoader): dataloader for the source labeled_data.
            dl_sup_t (data.DataLoader): dataloder for the target labeled data.
            dl_unsup_t (data.DataLoader): dataloder for the target unlabeled data.
            dl_val_t (data.DataLoader): dataloder for the target validation data.
        """
        self.args = args
        self.device = device
        self.directory = directory
        self.writer = writer
        self.dl_sup_s = dl_sup_s
        self.dl_sup_t = dl_sup_t
        self.dl_unsup_t = dl_unsup_t
        self.dl_val_t = dl_val_t
        self.ftr_ext = build_network(args).to(device)
        self.clf = build_classifier(args).to(device)
        self.es = EarlyStopping(args.patience, directory)


    def train(self):
        """Training module."""
        self.clf.apply(weights_init)

        params = []
        for key, value in dict(self.ftr_ext.named_parameters()).items():
            if value.requires_grad:
                if 'classifier' not in key:
                    params += [{'params': [value], 'lr': 0.1,
                                'weight_decay': 0.0005}]
                else:
                    params += [{'params': [value], 'lr': 0.1 * 10,
                                'weight_decay': 0.0005}]

        self.optimizer_f = SGD(params, lr=self.args.lr, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)
        self.optimizer_c = SGD(self.clf.parameters(), lr=1.0, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)

        param_lr_f = []
        for param_group in self.optimizer_f.param_groups:
            param_lr_f.append(param_group["lr"])
        param_lr_c = []
        for param_group in self.optimizer_c.param_groups:
            param_lr_c.append(param_group["lr"])

        for iteration in range(self.args.num_iterations):
            if iteration % len(self.dl_sup_s) == 0:
                iter_sup_s = iter(self.dl_sup_s)
            if iteration % len(self.dl_sup_t) == 0:
                iter_sup_t = iter(self.dl_sup_t)
            if iteration % len(self.dl_unsup_t) == 0:
                iter_unsup_t = iter(self.dl_unsup_t)
            x_s, y_s = next(iter_sup_s)
            x_sup_t, y_sup_t = next(iter_sup_t)
            x_unsup_t, _ = next(iter_unsup_t)
            
            x = torch.cat((x_s, x_sup_t), dim=0)
            y = torch.cat((y_s, y_sup_t), dim=0)
            x = x.float().to(self.device)
            y = y.long().to(self.device)
            x_sup_t = x_sup_t.float().to(self.device)
            y_sup_t = y_sup_t.long().to(self.device)
            x_unsup_t = x_unsup_t.float().to(self.device)

            self.optimizer_f = inv_lr_scheduler(param_lr_f, self.optimizer_f, iteration,
                                   init_lr=self.args.lr)
            self.optimizer_c = inv_lr_scheduler(param_lr_c, self.optimizer_c, iteration,
                                   init_lr=self.args.lr)

            self.ftr_ext.train()
            self.clf.train()
            self.zero_grad()

            #############################
            # CrossEntropy Minimization #
            #############################
            x_out = self.ftr_ext(x)
            pred_s = self.clf(x_out)
            l_xent = F.cross_entropy(pred_s, y)

            l_xent.backward(retain_graph=True)
            self.optimizer_f.step()
            self.optimizer_c.step()

            ################
            # Minimax Loss #
            #################
            self.zero_grad()
            x_out = self.ftr_ext(x_unsup_t)
            pred_u = self.clf(x_out, grad_rev=True)
            l_t = self.adentropy(pred_u)

            l_t.backward()
            self.optimizer_f.step()
            self.optimizer_c.step()

            if iteration%self.args.report_every==0:
                pred = np.argmax(pred_s.detach().cpu().numpy(), axis=1)
                metrics = {'Accuracy': accuracy_score(y.cpu().numpy(), pred),
                           'Total Loss': l_xent.item()}
                print('Training... Iteration {}'.format(iteration))
                self.print_and_log(metrics, 'train', iteration)
                stop = self.validate(iteration)
            if stop or iteration>=self.args.num_iterations:
                break
        
        self.load_weights()
        print('Best parameters restored.')
        self.writer.close()
        print("##########################################")

    def validate(self, iteration):
        """Validation module."""
        labels = []
        predictions = []
        total_loss = 0
        self.ftr_ext.eval()
        self.clf.eval()
        
        with torch.no_grad():
            for x, y in self.dl_val_t:
                x = x.float().to(self.device)
                y = y.long().to(self.device)

                # Computing the reconstruction score for each datapoint.
                x_out = self.ftr_ext(x)
                pred = self.clf(x_out)

                total_loss+=F.cross_entropy(pred, y).item()
                predictions.append(pred.detach().cpu())
                labels.append(y.cpu())
                
        predictions = np.argmax(torch.cat(predictions).numpy(), axis=1)
        labels = torch.cat(labels).numpy()
        total_loss /= len(self.dl_val_t)
        accuracy = accuracy_score(labels, predictions)

        metrics = {'Accuracy': accuracy,
                    'Total Loss': total_loss}
        print('Validation... Iteration {}'.format(iteration))
        self.print_and_log(metrics, 'val', iteration)
        #Early stopping checkpoint.
        stop, is_best = self.es.count(self.ftr_ext, self.clf, accuracy)
        if is_best:
            save_metrics(metrics, self.directory)
        return stop

    def adentropy(self, out):
        """Adentropy loss for the MME minimax optimization."""
        out = F.softmax(out, dim=1)
        adentropy = self.args.lambda_ * torch.mean(torch.sum(out * torch.log(out + 1e-10), dim=1))
        return adentropy

    def zero_grad(self):
        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()

    def load_weights(self):
        """Load the trained parameters of the model."""
        state_dict = torch.load('{}/weights/trained_parameters.pth'.format(self.directory))
        self.ftr_ext.load_state_dict(state_dict['feature_extractor'])
        self.clf.load_state_dict(state_dict['classifier'])

    def print_and_log(self, metrics, mode, epoch):
        for metric, value in metrics.items():
            print("{}: {:.3f}".format(metric, value))
            self.writer.add_scalar('{}_{}'.format(metric,mode), value, epoch)
        print("##########################################")
