import argparse
from utils import *
import time
from torch import nn
from torch import optim
from torchvision import models


def arg_parser():
    """
    parsers received arguments
    :return: parsed arguments
    """
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--data_dir', action='store')
    parser.add_argument('--arch', dest="arch", action="store", default="vgg16", choices=['vgg13', 'vgg16'])
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
    parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.01)
    parser.add_argument('--hidden_units', dest="hidden_units", action="store", default=1024)
    parser.add_argument('--epochs', dest="epochs", action="store", default=5)
    parser.add_argument('--gpu', dest="gpu", action="store_true", default=True)
    return parser.parse_args()


def train(model, criterion, optimizer, dataloaders, epochs, gpu):
    """
    train model based on input parameters on the given data
    :param model: loaded model
    :param criterion: defined criterion
    :param optimizer: defined optimizer
    :param dataloaders: loaded data with torch
    :param epochs: number of epochs
    :param gpu: cuda True or False
    :return: trained model and showing progress per epoch
    """
    start = time.time()

    # Use GPU if it's available
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")

    model.to(device)

    steps = 0
    running_loss = 0

    for epoch in range(epochs):
        epoch_time = time.time()
        for inputs, labels in dataloaders['train']:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            valid_loss = 0
            accuracy = 0
            model.eval()

            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                for inputs, labels in dataloaders['valid']:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)

                    valid_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            elapsed_epoch = time.time() - epoch_time
            print(f"Epoch {epoch+1}/{epochs}.. ",
                  f"Train loss: {running_loss/len(dataloaders['train']):.3f}.. ",
                  f"Validation accuracy: {accuracy/len(dataloaders['train']):.3f}.. ",
                  f"Validation loss: {valid_loss/len(dataloaders['valid']):.3f}.. ",
                  f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}.. ",
                  "time: {:.0f}m {:.0f}s".format(round(elapsed_epoch // 60, 2), round(elapsed_epoch % 60, 2)))

            running_loss = 0
            model.train()

    elapsed_time = time.time() - start
    print("\nTotal time: {:.0f}m {:.0f}s".format(elapsed_time//60, elapsed_time % 60))


def save_checkpoint(model, image_dataset, epochs, optimizer):
    """
    saves checkpoint
    :param model: loaded model
    :param image_dataset: loaded image dataset
    :param epochs: number of epochs
    :param optimizer: defined optimizer
    :return: saves checkpoint and prints message
    """
    # Saving the checkpoint
    model.class_to_idx = image_dataset['train'].class_to_idx

    checkpoint = {'model': model,
                  'input_size': 25088,
                  'output_size': 102,
                  'classifier': model.classifier,
                  'epochs': epochs,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'optimizer': optimizer.state_dict()}

    torch.save(checkpoint, 'checkpoint.pth')
    print("File Saved Successfully")


def main():
    args = arg_parser()
    gpu = args.gpu
    epochs = int(args.epochs)

    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    directories = (train_dir, valid_dir, test_dir)

    dataloaders = data_loader(directories)

    model = getattr(models, args.arch)(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = classifier_archs(args.arch, model, args.hidden_units)
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=float(args.learning_rate))
    class_index = image_datasets(directories)['train'].class_to_idx
    train(model, criterion, optimizer, dataloaders, epochs, gpu)
    model.class_to_idx = class_index
    save_checkpoint(model, image_datasets(directories), epochs, model.classifier)


if __name__ == "__main__":
    main()
