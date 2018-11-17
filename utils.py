#!/usr/bin/env python3


"""
Name:   utils.py
Author: Joshua Tice
Date:   November 10, 2018

Description
-----------
This file contains classes and fuctions that support both  train.py and predict.py,
including:

Classifier          Class used for instantiating the fully-connected classifier
                    portion of the neural network model
build_model         Function that handles the contruction of a neural network
load_checkpoint     Function that contructs a neural network from a checkpoint
load_data           Function that creates an interface for training, validation,
                    and testing data
preprocess_image    Function for processing an image such that it can be
                    classified by the neural network

Required packages
-----------------
numpy
skimage
torch
torchvision
utils.py
"""


import json
import numpy as np
import pprint
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


class Classifier(nn.Module):
    """Neural network classifier contructor"""

    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        """
        Parameters
        ----------
        input_size : int
            Number of input nodes
        output_size : int
            Number of output nodes
        hidden_layers : list of ints
            Number of nodes in each hidden layer of the classifier
        drop_p : float, optional
            Dropout of the classifier (between 0.0 and 1.0)
        """

        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        """
        Operations defining the forward pass of the neural network

        Parameters
        ----------
        x : torch tensor
            Tensor of images

        Returns
        -------
        array-like
            vector of predictions
        """

        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)

        x = self.output(x)

        return F.log_softmax(x, dim=1)


def build_model(base_model='vgg16', input_size=25088, output_size=102,
                hidden_layers=[4096, 1000], drop_p=0.5):
    """
    Loads a pre-trained convolutional neural network and configures a fully-
    connected classifier

    Parameters
    ----------
    base_model : str, optional
        The convolutional neural network the model will be based on.
        {'vgg16', 'vgg13'}
    input_size : int, optional
        Number of input nodes
    output_size : int, optional
        Number of output nodes
    hidden_layers : list of ints, optional
        Number of nodes in each hidden layer of the classifier
    drop_p : float, optional
        Dropout of the classifier (between 0.0 and 1.0)

    Returns
    -------
    torch convolutional neural network
    """

    if base_model == 'vgg11':
        model = models.vgg11(pretrained=True)
    elif base_model == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif base_model == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        raise ValueError(
            "base_model can only accept values in {'vgg11', 'vgg13', 'vgg16'}")

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = Classifier(input_size, output_size, hidden_layers, drop_p)

    return model


def load_checkpoint(filepath):
    """
    Instantiates and configures a torch neural network based on the parameters
    of a given checkpoint.

    Parameters
    ----------
    filepath : str
        File path to a saved checkpoint

    Returns
    -------
    torch neural network
        Instantiated network configured with checkpoint parameters
    """

    checkpoint = torch.load(filepath)
    model = build_model(
        base_model=checkpoint['model'],
        input_size=checkpoint['classifier_input_size'],
        output_size=checkpoint['classifier_output_size'],
        hidden_layers=checkpoint['classifier_hidden_layers'],
        drop_p=checkpoint['classifier_dropout'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_index = checkpoint['class_to_index']

    return model


def load_data(data_dir):
    """
    Loads data from data directory for neural network training

    Parameters
    ----------
    data_dir : str
        Path to the data directory containing images for training

    Returns
    -------
        dict
            Dictionary with training, validation, and testing data loaders in
            addition to dictionary with class to index relationship
    """
    training_dir = data_dir + '/train'
    validation_dir = data_dir + '/valid'
    testing_dir = data_dir + '/test'

    means = (0.485, 0.456, 0.406)
    std_devs = (0.229, 0.224, 0.225)

    training_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=30, expand=True),
        transforms.RandomResizedCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(means, std_devs),
    ])
    validation_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(means, std_devs),
    ])
    testing_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(means, std_devs),
    ])

    training_data = datasets.ImageFolder(
        training_dir, transform=training_transforms)
    validation_data = datasets.ImageFolder(
        validation_dir, transform=validation_transforms)
    testing_data = datasets.ImageFolder(
        testing_dir, transform=testing_transforms)

    training_loader = torch.utils.data.DataLoader(
        training_data, batch_size=64, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(
        validation_data, batch_size=32)
    testing_loader = torch.utils.data.DataLoader(
        testing_data, batch_size=32)

    data = {
        'training_loader': training_loader,
        'validation_loader': validation_loader,
        'testing_loader': testing_loader,
        'class_to_index': training_data.class_to_idx
    }

    return data


def preprocess_image(image_path):
    '''
    Scales, crops, and normalizes a PIL image for a PyTorch model

    Parameters
    ----------
    image_path : str
        Path to the image to be pre-processed

    Returns
    -------
    numpy.array
        Processed numpy image
    '''

    image = Image.open(image_path)

    # Scale image
    scaling_factor = 256 / min(image.size)
    scaled_size = (int(image.size[0] * scaling_factor),
                   int(image.size[1] * scaling_factor))
    image = image.resize(scaled_size, Image.ANTIALIAS)

    # Crop image
    box = ((int(image.size[1] / 2 - 112),
            int(image.size[0] / 2 - 112),
            int(image.size[1] / 2 + 112),
            int(image.size[0] / 2 + 112),))
    image = image.crop(box)

    # Normalize image
    np_image = np.array(image, dtype=float)
    means = np.array([0.485, 0.456, 0.406])
    std_devs = np.array([0.229, 0.224, 0.225])
    np_image /= 255
    np_image = (np_image - means) / std_devs

    # Transpose image for pytorch model
    np_image = np_image.transpose((2, 0, 1))

    return np_image
