import os
import torch
import gzip
import pickle
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from utils.data_from_list import data_fromlist, load_img



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


#############################
# Office/DomainNet Datasets #
#############################
class GetData(data.Dataset):
    def __init__(self, img_paths, domain, crop_size, mode):
        self.transforms =  {'train': transforms.Compose([
                                transforms.Resize(256),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(crop_size),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ]),
                            'val': transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(crop_size),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ]),
                            'test': transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(crop_size),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])
                             }
        self.base_path = './data/{}'.format(domain)
        self.x, self.y = data_fromlist(img_paths)
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
        x = load_img(os.path.join(self.base_path, x))
        x = self.transforms[self.mode](x)
        y = self.y[index]
        return x, y


def get_dataset(args):
    txt_path = './data/txt/{}'.format(args.domain)
    
    txt_file_s = os.path.join(txt_path,
                     'labeled_source_images_{}.txt'.format(args.source))
    
    txt_file_t = os.path.join(txt_path,
                     'labeled_target_images_{}_{}.txt'.format(args.target, args.n_shots))
                              
    txt_file_val = os.path.join(txt_path,
                     'validation_target_images_{}_{}.txt'.format(args.target, args.n_val))
                              
    txt_file_unl = os.path.join(txt_path,
                     'unlabeled_target_images_{}_{}.txt'.format(args.target, args.n_shots))

    if args.model_name == 'Alexnet':
        crop_size = 227
        bs = 32
    else:
        crop_size = 224
        bs = 24
    
    source_data = GetData(txt_file_s, args.domain, crop_size, mode='train')
    source_loader = torch.utils.data.DataLoader(source_data, 
                    batch_size=bs, shuffle=True, drop_last=True)
    
    target_data = GetData(txt_file_t, args.domain, crop_size, mode='train')
    target_loader = torch.utils.data.DataLoader(target_data,
                    batch_size=min(bs, len(target_data)), shuffle=True, drop_last=True)
                              
    target_data_val = GetData(txt_file_val, args.domain, crop_size, mode='val')
    target_loader_val = torch.utils.data.DataLoader(target_data_val,
                        batch_size=min(bs, len(target_data_val)), shuffle=True, drop_last=True)
                              
    target_data_unl = GetData(txt_file_unl, args.domain, crop_size, mode='train')
    target_loader_unl = torch.utils.data.DataLoader(target_data_unl, 
                        batch_size=bs * 2, shuffle=True, drop_last=True)
                              
    target_data_test = GetData(txt_file_unl, args.domain, crop_size, mode='test')
    target_loader_test = torch.utils.data.DataLoader(target_data_test,
                         batch_size=bs * 2, shuffle=True, drop_last=True)
                         
    return source_loader, target_loader, target_loader_unl, target_loader_val, target_loader_test


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
        if args.source=='svhn':
            x_test, y_test = test.data.numpy(), test.targets
            x_train = np.repeat(x_train.reshape(-1, 28, 28, 1), 3, 3)
            x_test = np.repeat(x_test.reshape(-1, 28, 28, 1), 3, 3)
        
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
        if args.source=='svhn':
            x_train = np.repeat(x_train.reshape(-1, 28, 28, 1), 3, 3)
            x_test = np.repeat(x_test.reshape(-1, 28, 28, 1), 3, 3)
        
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
