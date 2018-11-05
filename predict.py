#!/usr/bin/env python3

import argparse
import json
import utils

def predict(image_path, model, topk=5):
    '''
    Predict the class (or classes) of an image using a trained
    deep learning model

    Parameters
    ----------
    image_path : str
        Path to the image to be classified
    model : torch object
        Neural network used for prediction
    topk : int, optional

    '''

    # TODO: Implement the code to predict the class from an image file
    device = 'cpu'
    model.to(device)
    model.eval()
    with torch.no_grad():
        image = process_image(image_path)
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



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help="path to image to be categorized by model")
    parser.add_argument('checkpoint', help="model checkpoint to use for prediction")
    parser.add_argument('--top_k', type=int, default=5,
                        help="number of top predictions to show")
    parser.add_argument('--category_names', help="file for interpreting output")
    parser.add_argument('--gpu', action='store_true', help="run inference with gpu")

    args = parser.parse_args()


if __name__ == '__main__':
    main()
