from torchvision import transforms, datasets
import torch
from torch import nn


def data_transforms():
    """
    creates train, valid, and test transformations for data
    :return: dictionary containing train, valid, and test transforms
    """
    # Defining transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    # Test and Validation have the same transforms
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    return {'train': train_transforms, 'valid': test_transforms, 'test': test_transforms}


def image_datasets(directories):
    """

    :param directories: a tuple of train, valid, test directories
    :return: dictionary containing train, valid, and test image datasets
    """
    train_dir, valid_dir, test_dir = directories
    data_transform = data_transforms()

    # Loading the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=data_transform['train'])
    valid_data = datasets.ImageFolder(valid_dir, transform=data_transform['valid'])
    test_data = datasets.ImageFolder(test_dir, transform=data_transform['test'])

    return {'train': train_data, 'valid': valid_data, 'test': test_data}


def data_loader(directories):
    """
    Loads image_datasets with DataLoader() into a final dictionary
    :param directories: a tuple of train, valid, test directories
    :return: dictionary containing train, valid, and test dataloaders
    """
    image_dataset = image_datasets(directories)

    # Using the image datasets and the transforms, defining the dataloaders
    trainloader = torch.utils.data.DataLoader(image_dataset['train'], batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(image_dataset['valid'], batch_size=32)
    testloader = torch.utils.data.DataLoader(image_dataset['test'], batch_size=32)

    return {'train': trainloader, 'valid': validloader, 'test': testloader}


def classifier_archs(arch, model, hidden_units):
    """
    returns
    :param arch: model architecture in (str)
    :param model: model
    :param hidden_units: number of hidden units (int)
    :return: newly created classifier
    """
    if arch == "vgg13":
        feature_num = model.classifier[0].in_features
        classifier = nn.Sequential(nn.Linear(feature_num, hidden_units),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.5),
                                   nn.Linear(hidden_units, 102),
                                   nn.LogSoftmax(dim=1))

    elif arch == "vgg16":
        feature_num = model.classifier[0].in_features
        classifier = nn.Sequential(nn.Linear(feature_num, hidden_units),
                                   nn.ReLU(),
                                   nn.Dropout(0.5),
                                   nn.Linear(hidden_units, 102),
                                   nn.LogSoftmax(dim=1))
    else:
        feature_num = model.classifier[0].in_features
        classifier = nn.Sequential(nn.Linear(feature_num, hidden_units),
                                   nn.ReLU(),
                                   nn.Dropout(0.5),
                                   nn.Linear(hidden_units, 102),
                                   nn.LogSoftmax(dim=1))
    return classifier
