import argparse
import json
import torch
import numpy as np
from PIL import Image, ImageOps


def arg_parser():
    """
    parsers received arguments
    :return: parsed arguments
    """
    parser = argparse.ArgumentParser(description="Prediction")
    parser.add_argument('--image_path', type=str, help='Path to the input image file', required=True)
    parser.add_argument('--checkpoint', type=str, help='Path to the model checkpoint file', default='checkpoint.pth')
    parser.add_argument('--top_k', type=int, help='Choose top K.', default=5)
    parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
    parser.add_argument('--gpu', dest="gpu", action="store", default=True)

    args = parser.parse_args()

    return args


def load_checkpoint(path):
    """
    loads checkpoint from directory
    :param path: path to checkpoint, default: checkpoint.pth
    :return: saved model
    """
    checkpoint = torch.load(path)

    # Loading the model and optimizer with the saved pretrained architecture
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model


def process_image(image):
    """ Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a Numpy array
    """

    img = Image.open(image)
    img = img.resize((256, 256))
    cropped_image = ImageOps.fit(img, (224, 224), centering=(0.5, 0.5))
    numpy_img = np.array(cropped_image)/255

    # Normalize each color channel
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    numpy_img = (numpy_img - mean)/std

    numpy_img = numpy_img.transpose(2, 0, 1)

    return numpy_img


def predict(model, device, args, cat_to_name):
    """ Predict the class (or classes) of an image using a trained deep learning model.
    """
    model.eval()  # turn off dropout

    # The image
    image = process_image(args.image_path)

    # tranfer to tensor
    image = torch.from_numpy(np.array([image])).float()

    # Moving to selected device CPU/GPU
    image = image.to(device)
    model.to(device)
    model.eval()

    logps = model.forward(image)

    ps = torch.exp(logps)
    top_ps, top_class = ps.topk(args.top_k, dim=1)

    probs = top_ps.tolist()[0]
    list_idx = top_class.tolist()[0]

    idx_mapping = {val: key for key, val in model.class_to_idx.items()}

    classes = [idx_mapping[item] for item in list_idx]
    flowers = [cat_to_name[item] for item in classes]

    return probs, classes, flowers


def print_probs(probs, flowers):
    """
    prints out predicted top probabilities along with labels ordered
    :param probs:
    :param flowers:
    :return:
    """
    for i, (prob, label) in enumerate(zip(flowers, probs)):
        print(f"{i+1}:",
              f"{label.title()}, Probability: {prob*100}%")
    print("\n")


def main():

    args = arg_parser()

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    model = load_checkpoint(args.checkpoint)

    # Use GPU if it's available
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    top_probs, top_labels, top_flowers = predict(model, device, args, cat_to_name)

    print_probs(top_flowers, top_probs)


if __name__ == '__main__':
    main()
