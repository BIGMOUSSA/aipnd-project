import argparse
import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
from utils import  process_image
#define a early stopping function to avoid overfitting
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0

def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    model.to(device)
    model.eval()
    image = process_image(image_path).to(device)
    inverted_class = {k:v for v,k in model.class_to_idx.items()}
    with torch.no_grad():
      logps = model(image)
      ps = torch.exp(logps)
      top_p, top_class = ps.topk(topk, dim = 1)
      top_p = top_p.cpu().numpy()[0]
      top_class = top_class.cpu().numpy()[0]
      class_out = [inverted_class[k] for k in  top_class]
    return top_p, class_out