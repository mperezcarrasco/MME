import os
import json
import torch
import argparse
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from sklearn.metrics import accuracy_score

from utils.utils import seed_everything, save_metrics

from models.main import build_network, build_classifier
from preprocess import get_digits, get_dataset


def evaluate(ftr_extractor, classifier, device, dataloader, directory, mode, args):
    """Evaluation module.
    
        Args:
            model: (torch.nn): Model to be used for the evaluation.
            device: 'cuda' if available. 'cpu' else.
            dataloader (torch.data.DataLoader): iterator that contains the data to be evaluated.
            directory (str): Directory of the experiment.
            mode: 'val' if evaluating validation dataset, 'test' if evaluating test set.
            args: hyperparameters for the experiment.
    """
    print("Testing... ")

    criterion = nn.CrossEntropyLoss()

    labels = []
    predictions = []
    total_loss = 0
    ftr_extractor.eval()
    classifier.eval()
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.float().to(device)
            y = y.long().to(device)

            # Computing the reconstruction score for each datapoint.
            x_out = ftr_extractor(x)
            pred = classifier(x_out)

            total_loss+=criterion(pred, y).item()
            predictions.append(pred.detach().cpu())
            labels.append(y.cpu())
            
    predictions = np.argmax(torch.cat(predictions).numpy(), axis=1)
    labels = torch.cat(labels).numpy()
    total_loss /= len(dataloader)

    metrics = {'Accuracy': accuracy_score(labels, predictions)*100,
                'Total Loss': total_loss}
    print_metrics(metrics)
    save_metrics(metrics, directory, mode)
    return metrics

def print_metrics(metrics):
    for metric, value in metrics.items():
        print("{}: {:.2f}".format(metric, value))
    print("##########################################")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Size of each mini-batch.")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate for the Adam optimizer.")
    parser.add_argument("--temperature", type=float, default=0.05,
                        help="Hyperparameter temperature.")
    parser.add_argument("--domain", type=str, default='digits',
                         choices=['digits', 'office'],
                         help="Domain from which the dataset belongs.")
    parser.add_argument("--source", type=str, default='mnist',
                         choices=['mnist', 'usps', 'svhn', 'webcam', 'amazon', 'dslr'],
                         help="Dataset to be used as source for the experiment.")
    parser.add_argument("--target", type=str, default='mnist',
                         choices=['mnist', 'usps', 'svhn', 'webcam', 'amazon', 'dslr'],
                         help="Dataset to be used as target for the experiment.")
    parser.add_argument("--n_shots", type=int, default=3,
                         help="Number of labeled samples to be used for supervised training.")
    parser.add_argument("--n_val", type=int, default=3,
                         help="Number of labeled samples to be used for validation.")
    parser.add_argument("--model_name", type=str, default='AlexnetDigits',
                         choices=['Alexnet', 'Alexnetbn', 'VGG16', 'VGG16bn', 'AlexnetDigits'],
                         help="Name of the autoencoder model to be used for the experiment.")
    parser.add_argument("--pretrain", type=bool, default=True,
                         help="If domain is not digits, if the model must be pretrained on ImageNet.")
    args = parser.parse_args() 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything()

    # Path to store the results.
    parent_dir = 'results'
    job_name = '{}_{}_{}_{}_{}shots'.format(args.domain, args.source, args.target,
               args.model_name, args.n_shots)
    directory = os.path.join(parent_dir, job_name)

    if args.domain == 'office':
        args.n_classes=31
        _, _, _, _, dataloader_test = get_dataset(args)
    elif args.domain == 'digits':
        args.n_classes=10
        _, _, _, _, dataloader_test = get_digits(args)
    elif args.domain == 'multi':
        args.n_classes=126
        _, _, _, _, dataloader_test = get_dataset(args)

    # Call the model.
    model = build_network(args).to(device)
    classifier = build_classifier(args).to(device)

    # Restore the best model.
    weights_dir = os.path.join(directory, 'weights')
    state_dict = torch.load('{}/trained_parameters.pth'.format(weights_dir))
    model.load_state_dict(state_dict['feature_extractor'])
    classifier.load_state_dict(state_dict['classifier'])

    # The number of classes per root level must be a torch tensor.
    evaluate(model, classifier, device, dataloader_test, directory, 'test', args)
