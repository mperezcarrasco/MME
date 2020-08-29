from models.alexnet import Alexnet
from models.vgg16 import VGG16
from models.alexnetdigits import AlexnetDigits
from models.classifier import Classifier

def build_network(args):
    """Builds the feature extractor network for the MME.

        Args:
            args: Hyperparameters for the network building.

        Returns:
            model (torch.nn.Module): Network architecture.
    """
    # Checking if the network is implemented.
    implemented_networks = ('Alexnet', 'VGG16', 'AlexnetDigits')
    assert args.model_name in implemented_networks

    model = None

    if args.model_name == 'Alexnet':
        model = Alexnet(args)

    elif args.model_name == 'VGG16':
        model = VGG16(args)

    elif args.model_name == 'AlexnetDigits':
        model = AlexnetDigits(args)

    return model

def build_classifier(args):
    """Builds the classifier for the MME.

        Args:
            args: Hyperparameters for the classifier building.

        Returns:
            classifier (torch.nn.Module): Network architecture.
    """
    # Checking if the network is implemented.
    implemented_networks = ('Alexnet', 'VGG16', 'AlexnetDigits')
    assert args.model_name in implemented_networks

    if args.model_name == 'AlexnetDigits':
        classifier = Classifier(num_class=args.n_classes, inc=64, temp=args.temperature)
    else:
        classifier = Classifier(num_class=args.n_classes, inc=4096, temp=args.temperature)

    return classifier

