import os
import torch
import argparse

from utils.utils import seed_everything
from preprocess import get_digits, get_dataset
from train import Trainer
from test import evaluate

from torch.utils.tensorboard import SummaryWriter

def create_dirs(root_dir, dirs_to_create):
    """create directories to save metrics, plots, weights, etc...
    Args:
        root_dir (str): Root directory which contains all the experimentation.
        dirs_to_create(list): list of directories to create inside the root_dir.
    """
    for dir_ in dirs_to_create:
        dir_path = os.path.join(root_dir, dir_)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iterations", type=int, default=50000,
                        help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Size of each mini-batch.")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate for the Adam optimizer.")
    parser.add_argument("--patience", type=int, default=5,
                        help="Patience for the early stopping.")
    parser.add_argument("--latent_dim", type=int, default=32,
                        help="Dimension of the latent space of the Autoencoder.")
    parser.add_argument("--temperature", type=float, default=0.05,
                        help="Hyperparameter temperature.")
    parser.add_argument("--lambda_", type=float, default=0.1,
                        help="Hyperparameter lambda.")
    parser.add_argument("--domain", type=str, default='digits',
                         choices=['digits', 'office', 'multi'],
                         help="Domain from which the dataset belongs.")
    parser.add_argument("--source", type=str, default='mnist',
                         choices=['mnist', 'usps', 'svhn', 'webcam', 'amazon', 'dslr', 'real', 'sketch', 'painting', 'clipart'],
                         help="Dataset to be used as source for the experiment.")
    parser.add_argument("--target", type=str, default='mnist',
                         choices=['mnist', 'usps', 'svhn', 'webcam', 'amazon', 'dslr', 'real', 'sketch', 'painting', 'clipart'],
                         help="Dataset to be used as target for the experiment.")
    parser.add_argument("--n_shots", type=int, default=3,
                         help="Number of labeled samples to be used for the target.")
    parser.add_argument("--n_val", type=int, default=3,
                         help="Number of labeled samples to be used for validation.")
    parser.add_argument("--model_name", type=str, default='AlexnetDigits',
                         choices=['Alexnet', 'VGG16', 'AlexnetDigits'],
                         help="Name of the model to be used for the experiment.")
    parser.add_argument("--pretrain", type=bool, default=True,
                         help="If domain is not digits, if the model must be pretrained on ImageNet.")
    parser.add_argument("--report_every", type=int, default=500,
                        help="Number of iterations from which the metrics must be reported.")
    args = parser.parse_args()

    seed_everything()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create dir to store the results.
    parent_dir = 'results'
    job_name = '{}_{}_{}_{}_{}shots'.format(args.domain, args.source, args.target,
               args.model_name, args.n_shots)
    directory = os.path.join(parent_dir, job_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # create weights dir.
    create_dirs(directory, ['weights'])

    # Tensorboard logger
    writer = SummaryWriter(directory)

    #Create all the dataloaders.
    if args.domain == 'office':
        args.n_classes=31
        dataloader_source, dataloader_sup, dataloader_unsup, \
        dataloader_val, dataloader_test = get_dataset(args)
    elif args.domain == 'digits':
        args.n_classes=10
        dataloader_source, dataloader_sup, dataloader_unsup, \
         dataloader_val, dataloader_test = get_digits(args)
    elif args.domain == 'multi':
        args.n_classes=126
        dataloader_source, dataloader_sup, dataloader_unsup, \
         dataloader_val, dataloader_test = get_dataset(args)


    mme = Trainer(args,
                device,
                writer,
                directory,
                dataloader_source,
                dataloader_sup,
                dataloader_unsup,
                dataloader_val)
    mme.train()
    evaluate(mme.ftr_ext, mme.clf, device, dataloader_test, directory, 'test', args)
