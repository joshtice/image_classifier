#!/usr/bin/env python3


"""
Name:   predict.py
Author: Joshua Tice
Date:   November 10, 2018

Description
-----------
This script takes an image of a flower and makes a prediction of the flower's
class. The script reqires a checkpoint that defines the architecture of a
convolutional neural network.

Usage
-----
Basic usage:
python predict.py image_path checkpoint

Options:
--top_k             Top number of predictions to output
--category_names    File to convert predictions to human-readable form
--gpu               Whether or not to use gpu for inference

Required packages
-----------------
numpy
skimage
torch
torchvision
utils.py

To do
-----
- Fix gpu issue
- Add some logging for troubleshooting
- the name function isn't working
"""


import argparse
import json
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
from torchvision import datasets, models, transforms
import utils


def parse_args():
    """
    Parses arguments from the command line

    Returns
    -------
    argparse.Namespace object
        Container with parsed arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', 
        help="path to image to be categorized by model")
    parser.add_argument('checkpoint', 
        help="model checkpoint to use for prediction")
    parser.add_argument('--top_k', type=int,
        help="number of top predictions to show")
    parser.add_argument('--category_names', 
        help="file for interpreting output")
    parser.add_argument('--gpu', action='store_true', 
        help="run inference with gpu")
    args = parser.parse_args()

    if not os.path.isfile(args.image_path):
        raise ValueError("image_path does not exist")

    if not os.path.isfile(args.checkpoint):
        raise ValueError("checkpoint does not exist")

    if args.top_k is not None:
        if args.top_k < 1:
            raise ValueError("top_k must be greater than 0")
    else:
        args.top_k = 1

    if ((args.category_names is not None) and 
       (not os.path.isfile(args.category_names))):
        raise ValueError("category_names does not exist")

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
        Indicates whether to run inference using cpu or gpu. 
        Allowed values: {'cpu', 'gpu'}
    '''

    model.to(device)
    model.eval()
    with torch.no_grad():
        image = utils.preprocess_image(image_path)
        image = np.expand_dims(image, axis=0)
        image = torch.tensor(image).float()
        image = image.to(device)
        pred = model.forward(image)
        probs, classes = pred.topk(topk)
        probs = torch.exp(probs)
#     print(probs)
#     print(classes)
    probs = probs.to('cpu').numpy().tolist()[0]
    classes = classes.to('cpu').numpy().tolist()[0]
    for key, value in model.class_to_index.items():
        classes[classes == value] = int(key)

    return probs, classes


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
    classes = list(map(str, classes))
    names = list(map(category_mapper.get, classes))
    names = [x if x is not None else 'name not available' for x in names ]

    return names


def main():
    print("parsing arguments...")
    args = parse_args()

    device = torch.device(
        "cuda:0" if (args.gpu and torch.cuda.is_available()) else "cpu")
    print("loading model on device {}...".format(device))
    model = utils.load_checkpoint(args.checkpoint)

    print("running prediction...")
    probs, classes = predict(args.image_path, model, 
                             topk=args.top_k, device=device)
    
    if args.category_names is not None:
        print("translating results...")
        classes = translate_classes(classes, args.category_names)
        
    print("Top predictions:")
    print("Class                    Probability")
    print("-----                    -----------")
    for name, prob in zip(classes, probs):
        print("{:<25}{:<11.3f}".format(name, prob))


if __name__ == '__main__':
    main()
