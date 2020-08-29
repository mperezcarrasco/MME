import torch
from torch.optim import SGD
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from models.main import build_network, build_classifier
from sklearn.metrics import accuracy_score
from utils.utils import EarlyStopping, weights_init, save_metrics
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
        self.optimizer_f = SGD(self.ftr_ext.parameters(), lr=self.args.lr, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)
        self.optimizer_c = SGD(self.clf.parameters(), lr=1.0, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)

        # Some images to be reconstructed every 50 epochs.
        iteration=0
        while iteration < self.args.num_iterations:
            for (x_s, y_s), (x_unsup_t,_), (x_sup_t, y_sup_t) in \
                zip(cycle(self.dl_sup_s), cycle(self.dl_unsup_t), cycle(self.dl_sup_t)):
                # Converting features to FloatTensor and to device.
                x = torch.cat((x_s, x_sup_t), dim=0)
                y = torch.cat((y_s, y_sup_t), dim=0)
                x = x.float().to(self.device)
                y = y.long().to(self.device)
                x_sup_t = x_sup_t.float().to(self.device)
                y_sup_t = y_sup_t.long().to(self.device)
                x_unsup_t = x_unsup_t.float().to(self.device)

                self.ftr_ext.train()
                self.clf.train()
                self.zero_grad()

                #############################
                # CrossEntropy Minimization #
                #############################
                x_out = self.ftr_ext(x)
                pred = self.clf(x_out)
                l_xent = F.cross_entropy(pred, y)

                l_xent.backward(retain_graph=True)
                self.optimizer_f.step()
                self.optimizer_c.step()

                ################
                # Minimax Loss #
                #################
                self.zero_grad()
                x_out = self.ftr_ext(x_unsup_t)
                pred = self.clf(x_out)
                l_mm = self.adentropy(x_out)

                l_mm.backward()
                self.optimizer_f.step()
                self.optimizer_c.step()

                if iteration%self.args.report_every==0:
                    pred = np.argmax(pred.detach().cpu().numpy(), axis=1)
                    metrics = {'Accuracy': accuracy_score(y.cpu().numpy(), pred),
                               'Total Loss': l_xent.item()}
                    self.print_and_log(metrics, 'train', iteration)
                    stop = self.validate(iteration)
                iteration+=1
                if stop or iteration>=self.args.num_iterations:
                    iteration=9999999
                    break
        
        self.load_weights()
        print('Best parameters restored.')
        self.writer.close()
        print("##########################################")

    def validate(self, epoch):
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
        self.print_and_log(metrics, 'val', epoch)
        #Early stopping checkpoint.
        stop, is_best = self.es.count(self.ftr_ext, self.clf, accuracy)
        if is_best:
            save_metrics(metrics, self.directory)
        return stop

    def adentropy(self, out):
        """Adentropy loss for the MME minimax optimization."""
        out = F.softmax(out, dim=1)
        adentropy = -self.args.lambda_ * torch.mean(torch.sum(out * torch.log(out + 1e-10), dim=1))
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
            #print("{}: {:.3f}".format(metric, value))
            self.writer.add_scalar('{}_{}'.format(metric,mode), value, epoch)
        #print("##########################################")
