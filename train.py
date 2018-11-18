#!/usr/bin/env python3

"""
Name:   train.py
Author: Joshua Tice
Date:   November 18, 2018

Description
-----------
This script trains a neural network on a directory of images. Tne neural network
consists of multiple pre-trained convolutional layers with an architecture
developed by the Visual Geometry Group at Oxford. The user can append a custom,
fully-connected classifier to the backend of the convolutional layers and
specify certain hyperparameters. After training, the script saves a checkpoint
of the model.

Usage
-----
Basic usage:
python train.py directory_with_images

Options:
--save_dir directory    Directory where checkpoint will be saved
--arch                  Convolutional layer architecture to use
                        {'vgg11', 'vgg13', 'vgg16'}, default='vgg16'
--hidden_units          Comma-separated number of nodes in hidden layers
                        of classifier
--dropout               Dropout value to use for classifier
--learning_rate         Learning rate during training
--epochs                Number of epochs to train through
--gpu                   Indicates whether to use GPU if available

Required packages
-----------------
numpy
skimage
torch
torchvision
utils.py
"""


import argparse
import datetime
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


DEFAULTS = {
    'save_dir': os.getcwd(),
    'arch': 'vgg16',
    'hidden_units': [4096, 1000],
    'dropout': 0.5,
    'learning_rate': 0.001,
    'epochs': 3,
}


def parse_args(defaults=DEFAULTS):
    """
    Parse arguments from the command line

    Returns
    -------
    argparse.Namespace object
        Container with parsed arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory',
                        help="directory containing images for training")
    parser.add_argument('--save_dir',
                        help="directory where checkpoint will be saved")
    parser.add_argument('--arch', choices=['vgg11', 'vgg13', 'vgg16'],
                        help="convolutional neural network architecture to implement")
    parser.add_argument('--hidden_units',
                        help="nodes in classifier inner layers, comma-separated")
    parser.add_argument('--dropout', type=float,
                        help="classifier dropout")
    parser.add_argument('--learning_rate', type=float,
                        help="learning rate")
    parser.add_argument('--epochs', type=int,
                        help="number of epochs to train the model on")
    parser.add_argument('--gpu', action='store_true',
                        help="train model using gpu")
    args = parser.parse_args()

    if args.save_dir:
        if not os.path.isdir(args.save_dir):
            print("save path not valid")
            raise ValueError("Please enter a valid save directory")
    else:
        args.save_dir = defaults['save_dir']

    if args.arch is None:
        args.arch = 'vgg16'

    if args.hidden_units:
        try:
            args.hidden_units = [int(units) for units in args.hidden_units.split(',')]
        except:
            raise ValueError('Hidden units need to be entered as a comma-separated\
                string of integers')
    else:
        args.hidden_units = defaults['hidden_units']

    if args.dropout:
        if args.dropout < 0 or args.dropout > 1:
            raise ValueError("Dropout needs to be a value between 0.0 and 1.0")
    else:
        args.dropout = defaults['dropout']

    if args.learning_rate:
        if args.learning_rate < 0:
            raise ValueError("Learning rate must be greater than 0")
    else:
        args.learning_rate = defaults['learning_rate']

    if args.epochs:
        if args.epochs < 0:
            raise ValueError("Number of epochs must be greater than 0")
    else:
        args.epochs = defaults['epochs']

    return args


def validate(model, device, validation_loader, criterion):
    """
    Run validation on the model

    Parameters
    ----------
    model : pytorch model
        Fully instantiated and configured pytorch model
    device : str
        Device to use for validation: {'cpu', 'gpu'}
    validation_loader : pytorch data loader
        Images used for validation
    criterion : pytorch function
        Formula for evaluating model

    Returns
    -------
    tuple of floats
        Loss and accuracy calculated from the validation
    """

    model.eval()

    validation_loss = 0
    accuracy = 0

    with torch.no_grad():
        for inputs, labels in iter(validation_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model.forward(inputs)
            validation_loss += criterion(outputs, labels).item()
            ps = torch.exp(outputs)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()

    model.train()

    return validation_loss, accuracy


def train_model():
    """
    Train a neural network model on a directory of images. Output a model
    checkpoint after training.
    """

    # Parse arguments
    print("parsing arguments...")
    args = parse_args()

    # Build model architecture
    print("building model...")
    model = utils.build_model(
        base_model=args.arch,
        input_size=25088,
        output_size=102,
        hidden_layers=args.hidden_units,
        drop_p=args.dropout)

    # Send model to gpu/cpu
    device = torch.device("cuda:0" if args.gpu else "cpu")
    print("sending model to device {}...".format(device))
    model.to(device)

    # Load data
    print("loading data...")
    data = utils.load_data(args.data_directory)
    training_loader = data['training_loader']
    validation_loader = data['validation_loader']
    testing_loader = data['testing_loader']
    model.class_to_index = data['class_to_index']

    # Train classifier
    print("training model...")
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    print_every = 40
    steps = 0

    for epoch in range(args.epochs):
        running_loss = 0
        model.train()
        for inputs, labels in iter(training_loader):
            steps += 1
            print("step number {}...".format(steps))
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Run validation
            if steps % print_every == 0:
                validation_loss, accuracy = validate(
                    model, device, validation_loader, criterion)
                print("Epoch: {}/{}.. ".format(epoch + 1, args.epochs),
                      "Training Loss: {:.3f}.. ".format(
                    running_loss / print_every),
                    "Validation Loss: {:.3f}.. ".format(
                    validation_loss / len(validation_loader)),
                    "Validation Accuracy: {:.3f}".format(
                    accuracy / len(validation_loader)))
                running_loss = 0

    # Evaluate model on test data
    test_loss, test_accuracy = validate(model, device, testing_loader, criterion)
    print("Test Loss: {:.3f}.. ".format(test_loss / len(testing_loader)),
          "Test Accuracy: {:.3f}".format(test_accuracy / len(testing_loader)))

    # Save checkpoint
    print("saving checkpoint...")
    checkpoint = {
        'model': args.arch,
        'classifier_input_size': 25088,
        'classifier_output_size': 102,
        'classifier_hidden_layers': args.hidden_units,
        'classifier_dropout': args.dropout,
        'model_state_dict': model.state_dict(),
        'class_to_index': model.class_to_index,
        # 'training_epochs': args.epochs,
        # 'optimizer': optimizer,
        # 'learning_rate': args.learning_rate,
        # 'optimizer_state_dict': optimizer.state_dict(),
    }

    checkpoint_name = "{}_checkpoint.pth".format(
        datetime.datetime.now().strftime('%Y-%m-%d'))
    checkpoint_path = os.path.join(args.save_dir, checkpoint_name)
    torch.save(checkpoint, checkpoint_path)
    print("DONE!!!")


if __name__ == '__main__':
    train_model()
