#!/usr/bin/env python3

import argparse
import utils
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import skimage
import skimage.io
import skimage.transform
import skimage.util
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms, models


def parse_args():
    """
    Parses arguments from the command line

    Returns
    -------
    argparse.Namespace object
        Container with parsed arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help="path to image to be categorized by model")
    parser.add_argument('checkpoint', help="model checkpoint to use for prediction")
    parser.add_argument('--top_k', type=int, default=1,
                        help="number of top predictions to show")
    parser.add_argument('--category_names', help="file for interpreting output")
    parser.add_argument('--gpu', action='store_true', help="run inference with gpu")
    args = parser.parse_args()

    return args


def predict(image_path, model, topk=1, device='cpu'):
    '''
    Predict the class (or classes) of an image using a trained
    deep learning model

    Parameters
    ----------
    image_path : str
        Path to the image to be classified
    model : torch object
        Neural network used for prediction
    top_k : int, optional
        Number of top categories to print for prediction
    device : str
        Indicates whether to run inference using cpu or gpu. Allowed values:
        {'cpu', 'gpu'}
    '''

    model.to(device)
    model.eval()
    with torch.no_grad():
        image = utils.preprocess_image(image_path)
        image = np.expand_dims(image, axis=0)
        image = torch.tensor(image).float()
        image.to(device)
        pred = model.forward(image)
        probs, classes = pred.topk(topk)
        probs = torch.exp(probs)
    probs = probs.numpy()
    classes = classes.numpy()
    for key, value in model.class_to_index.items():
        classes[classes == value] = key

    return probs.squeeze(), classes.squeeze()


def translate_classes(classes, json_file):
    """
    Convert torch model outputs to human-readable categories

    Parameters
    ----------
    classes : array-like
        Numerical class output of the neural network
    json_file : str
        Path to json file with category/class mapping

    Returns
    -------
    list
        List of strings with human-readable predicted classes
    """

    with open(json_file, 'r') as f:
        category_mapper = json.load(f)
    names = list(map(category_mapper.get, list(classes.astype(str))))

    return names


def main():
    """
    Main script method
    """

    args = parse_args()
    device = torch.device(
        "cuda:0" if (args.gpu and torch.cuda.is_available()) else "cpu")
    model = utils.load_checkpoint(args.checkpoint)
    probs, classes = predict(args.image_path, model, topk=args.top_k, device=device)
    if args.category_names is not None:
        classes = translate_classes(classes, args.category_names)

    print("Top predictions:")
    for class, prob in zip(classes, probs):
        print("{} {}".format(class, prob))


if __name__ == '__main__':
    main()
