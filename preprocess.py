import torch
import gzip
import pickle
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


############################
# Get some labeled samples #
############################
def get_labeled_samples(x, y, n_samples):
    np.random.seed(14)
    classes = np.unique(y)
    indxs = [np.where(y == class_) for class_ in classes]

    ix = []
    for indx in indxs:
        ix.extend(np.random.choice(indx[0], n_samples, replace = False))

    np.random.shuffle(ix)
    x_sup = x[ix]
    y_sup = y[ix]
    return x_sup, y_sup, ix


##################
# Office Dataset #
##################
class OfficeData(data.Dataset):
    def __init__(self, x, y, mode):
        self.transforms =  {'train': transforms.Compose([
                                transforms.Resize(256),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ]),
                            'val': transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ]),
                            'test': transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])
                             }
        self.x = x
        self.y = y
        self.mode = mode
        
    def __len__(self):
        """
        Number of images in the object dataset.
        """
        return self.x.shape[0]

    def __getitem__(self, index):
        """
        Return an item from the dataset.
        """
        x = self.x[index]
        y = self.y[index]
        x = Image.fromarray(x)
        x = self.transforms[self.mode](x)
        return x, y

def get_office_dataset(args, domain):
    if domain=='source':
        data_dir='./data/office31/{}/images/'.format(args.source)
        data = datasets.ImageFolder(data_dir)

        x_train = np.array([np.array(data[i][0]) for i in range(len(data))])
        y_train = np.array([data[i][1] for i in range(len(data))])

        batch_size = min(args.batch_size, args.n_shots*args.n_classes)
        dataloader_train = DataLoader(OfficeData(x_train, y_train, 'train'), 
                                      batch_size=batch_size, shuffle=True, drop_last=True)
        return dataloader_train
    elif domain=='target':
        data_dir='./data/office31/{}/images/'.format(args.target)
        data = datasets.ImageFolder(data_dir)

        x_train = np.array([np.array(data[i][0]) for i in range(len(data))])
        y_train = np.array([data[i][1] for i in range(len(data))])
        
        x_val, y_val, ixs = get_labeled_samples(x_train, y_train, args.n_val)
        dataloader_val = DataLoader(OfficeData(x_val, y_val, 'val'), 
                         batch_size=args.batch_size*2, shuffle=False)
        
        x_train, y_train = np.delete(x_train, ixs, axis=0), np.delete(y_train, ixs, axis=0)

        batch_size = min(args.batch_size, args.n_shots*args.n_classes)
        x_sup, y_sup, ixs = get_labeled_samples(x_train, y_train, args.n_shots)
        dataloader_sup = DataLoader(OfficeData(x_sup, y_sup, 'train'), batch_size=batch_size, 
                                    shuffle=True, drop_last=True)

        x_train, y_train = np.delete(x_train, ixs, axis=0), np.delete(y_train, ixs, axis=0)

        dataloader_unsup = DataLoader(OfficeData(x_train, y_train, 'train'), 
                                      batch_size=args.batch_size*2, shuffle=True)

        #Same as unsupervised but shuffle=False.
        dataloader_test = DataLoader(OfficeData(x_train, y_train, 'test'), 
                                     batch_size=args.batch_size*2, shuffle=False)
        return dataloader_sup, dataloader_unsup, dataloader_val, dataloader_test

def get_office(args):
    dataloader_source = get_office_dataset(args, 'source')
    dataloader_sup, dataloader_unsup, dataloader_val, dataloader_test = get_office_dataset(args, 'target')
    return dataloader_source, dataloader_sup, dataloader_unsup, dataloader_val, dataloader_test


##################
# Digits Dataset #
##################
class DigitDataset(data.Dataset):
    """This class is needed to processing batches for the dataloader."""
    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.transform = transforms.Compose([transforms.Resize(32),
                                             transforms.ToTensor()])

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        x = Image.fromarray(x)
        x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)


def get_mnist(args, domain, data_dir='./data/mnist/'):
    train = datasets.MNIST(root=data_dir, train=True, download=True)
    test = datasets.MNIST(root=data_dir, train=False, download=True)

    x_train = train.data.numpy()
    y_train = train.targets

    if domain=='source':
        batch_size = min(args.batch_size, args.n_shots*args.n_classes)
        dataloader_source = DataLoader(DigitDataset(x_train, y_train), batch_size=batch_size, 
                                       shuffle=True, drop_last=True)
        return dataloader_source

    elif domain=='target':
        x_val, y_val, ixs = get_labeled_samples(x_train, y_train, args.n_val)
        dataloader_val = DataLoader(DigitDataset(x_val, y_val), batch_size=args.batch_size*2, 
                                    shuffle=False)
            
        x_train, y_train = np.delete(x_train, ixs, axis=0), np.delete(y_train, ixs, axis=0)

        batch_size = min(args.batch_size, args.n_shots*args.n_classes)
        x_sup, y_sup, _ = get_labeled_samples(x_train, y_train, args.n_shots)
        dataloader_sup = DataLoader(DigitDataset(x_sup, y_sup), batch_size=batch_size, 
                                    shuffle=True, drop_last=True)

        dataloader_unsup = DataLoader(DigitDataset(x_train, y_train), batch_size=args.batch_size*2, 
                                    shuffle=True)
        
        x_test = test.data.numpy()
        y_test = test.targets
        dataloader_test = DataLoader(DigitDataset(x_test, y_test), batch_size=args.batch_size*2, 
                                    shuffle=False)

        return dataloader_sup, dataloader_unsup, dataloader_val, dataloader_test

def get_usps(args, domain, data_dir='./data/usps'):
    with gzip.open('{}/usps_28x28.pkl'.format(data_dir), 'rb') as f:
        (x_train, y_train), (x_test, y_test) = pickle.load(f, encoding='bytes')
    x_train *= 255
    x_test *= 255
    x_train = x_train.reshape(-1, 28, 28).astype('uint8')
    x_test = x_test.reshape(-1, 28, 28).astype('uint8')

    if domain=='source':
        batch_size = min(args.batch_size, args.n_shots*args.n_classes)
        dataloader_source = DataLoader(DigitDataset(x_train, y_train), batch_size=batch_size, 
                                  shuffle=True, drop_last=True)
        return dataloader_source

    elif domain=='target':
        x_val, y_val, ixs = get_labeled_samples(x_train, y_train, args.n_val)
        dataloader_val = DataLoader(DigitDataset(x_val, y_val), batch_size=args.batch_size*2, 
                                    shuffle=False)
            
        x_train, y_train = np.delete(x_train, ixs, axis=0), np.delete(y_train, ixs, axis=0)

        batch_size = min(args.batch_size, args.n_shots*args.n_classes)
        x_sup, y_sup, _ = get_labeled_samples(x_train, y_train, args.n_shots)
        dataloader_sup = DataLoader(DigitDataset(x_sup, y_sup), batch_size=batch_size, 
                                    shuffle=True, drop_last=True)

        dataloader_unsup = DataLoader(DigitDataset(x_train, y_train), batch_size=args.batch_size*2, 
                                    shuffle=True)
        
        dataloader_test = DataLoader(DigitDataset(x_test, y_test), batch_size=args.batch_size*2, 
                                    shuffle=False)
        return dataloader_sup, dataloader_unsup, dataloader_val, dataloader_test


def get_svhn(args, domain, data_dir='./data/svhn/'):
    train = datasets.SVHN(root=data_dir, split='train', download=True)
    test = datasets.SVHN(root=data_dir, split='test', download=True)

    x_train = train.data.transpose(0,2,3,1)
    y_train = train.labels

    if domain=='source':
        batch_size = min(args.batch_size, args.n_shots*args.n_classes)
        dataloader_source = DataLoader(DigitDataset(x_train, y_train), batch_size=batch_size, 
                                  shuffle=True, drop_last=True)
        return dataloader_source

    elif domain=='target':
        x_val, y_val, ixs = get_labeled_samples(x_train, y_train, args.n_val)
        dataloader_val = DataLoader(DigitDataset(x_val, y_val), batch_size=args.batch_size*2, 
                                    shuffle=False)
        
        x_train, y_train = np.delete(x_train, ixs, axis=0), np.delete(y_train, ixs, axis=0)

        batch_size = min(args.batch_size, args.n_shots*args.n_classes)
        x_sup, y_sup, _ = get_labeled_samples(x_train, y_train, args.n_shots)
        dataloader_sup = DataLoader(DigitDataset(x_sup, y_sup), batch_size=batch_size, 
                                    shuffle=True, drop_last=True)

        dataloader_unsup = DataLoader(DigitDataset(x_train, y_train), batch_size=args.batch_size*2, 
                                    shuffle=True)
        
        x_test = test.data.transpose(0,2,3,1)
        y_test = test.labels
        dataloader_test = DataLoader(DigitDataset(x_test, y_test), batch_size=args.batch_size*2, 
                                    shuffle=False)

        return dataloader_sup, dataloader_unsup, dataloader_val, dataloader_test


def get_digits(args):
    if args.source=='mnist':
        dataloader_source = get_mnist(args, 'source')
    elif args.source=='svhn':
        dataloader_source = get_svhn(args, 'source')
    elif args.source=='usps':
        dataloader_source = get_usps(args, 'source')

    if args.target=='mnist':
        dataloader_sup, dataloader_unsup, dataloader_val, dataloader_test = get_mnist(args, 'target')
    elif args.target=='svhn':
        dataloader_sup, dataloader_unsup, dataloader_val, dataloader_test = get_svhn(args, 'target')
    elif args.target=='usps':
        dataloader_sup, dataloader_unsup, dataloader_val, dataloader_test = get_usps(args, 'target')
    return dataloader_source, dataloader_sup, dataloader_unsup, dataloader_val, dataloader_test
