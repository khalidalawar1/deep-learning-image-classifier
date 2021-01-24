import argparse

import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
import torch.nn.functional as F
from workspace_utils import *

from PIL import Image
import os
import numpy as np
import math
import matplotlib.pyplot as plt


def get_predict_input_args():
    Parse = argparse.ArgumentParser()
    Parse.add_argument('input')
    Parse.add_argument('checkpoint')
    Parse.add_argument('--top_k',default="3", type=int)
    Parse.add_argument('--category_names',default="cat_to_name.json")
    Parse.add_argument('--gpu',action="store_true", default=False)


    args = Parse.parse_args()
   

    return args

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = ''
    hidden_units = checkpoint['hidden_units']
    
    if checkpoint['architecture'] == "vgg16":
        model = models.vgg16(pretrained=True)
    elif checkpoint['architecture'] == "vgg13":
        model = models.vgg13(pretrained=True)
        
    # Freezing parameters so we don't backprop through them, this would have extensive computing overhead!
    for param in model.parameters():
        param.requires_grad = False

    #the pretrained model has 4096 in_features in the first classifier layer, that is why I am defining as 4096 as well.
    model.classifier = nn.Sequential(nn.Linear(model.classifier[0].in_features, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(hidden_units, int(hidden_units/2)),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(int(hidden_units/2), 102), #the final out_features is 102 because len(cat_to_name) is 102
                                     nn.LogSoftmax(dim=1))

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    img = Image.open(image)
    
    width, height = img.size #getting height and width of image
    
    if width >= height: #check if image is landscape
        scale = 256 / height # if the image is landscape then the shortest side is the height, therefore I am resizing the image so that the shortest side is 256 pixels just like the instructions stated
    else:
        scale = 256 / width
    
    
    scaled_width = int(width*scale)
    scaled_height =  int(height*scale)
    img = img.resize((scaled_width,scaled_height)) #resize done
    
    
    #crop to 224px as requested to get a square 224x224 center portion of the image
    left = (scaled_width - 224)/2
    top = (scaled_height - 224)/2
    right = (scaled_width + 224)/2
    bottom = (scaled_height + 224)/2
    
    #cropping function
    img = img.crop((left, top, right, bottom))
    
    np_image = np.array(img) #converting image from PIL to np for further pre processing
    np_image = (np_image - 0)/(255-0) #normalizing 0-255 values to be between 0-1 inclusive
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std #np_image fully pre processed
    

    np_image = np.transpose(np_image,(2,0,1)) #swap the color channels from the 3rd dimension with the 1st dimension for pytorch
    return torch.Tensor(np_image)

def predict(image_path, model, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    #defining the device to run the inference on a GPU if available, else use the CPU
    device=''
    if torch.cuda.is_available() & gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device);
    
    image = process_image(image_path)
    
    image.unsqueeze_(0)
    with torch.no_grad():
        model.eval()
        image = image.to(device)
        log_ps = model(image)
        ps = torch.exp(log_ps)
    top_ps, top_classes = ps.topk(topk, dim=1)
    return top_ps, top_classes
