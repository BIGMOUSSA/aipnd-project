
# import package
import argparse
import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
from utils import load_checkpoint, process_image
# definir arg parameters
parser = argparse.ArgumentParser(description = "Application to predict a image label")
parser.add_argument('image_path', help = 'the path to the image to predict')
parser.add_argument('checkpoint', help = 'path to the checkpoint model', default= "checkpoints/checkpoint.pt")
parser.add_argument('--top_k', help = "the number of top classe after prediction", type = int, default = 1)
parser.add_argument('--category-name', help = " a json file for the category name", default= "cat_to_name.json")
parser.add_argument('--gpu', default = "gpu")

args = parser.parse_args()

model, optimizer, num_epoch = load_checkpoint(args.checkpoint)
# lead file and make preprocessing
image = process_image(args.image_path)

inverted_class = {k:v for v,k in model.class_to_idx.items()}
#predict label
if args.gpu == "gpu" :
    # Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device('cpu')
    
def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    model.to(device)
    model.eval()
    image = process_image(image_path).to(device)
 
    with torch.no_grad():
      logps = model(image)
      ps = torch.exp(logps)
      top_p, top_class = ps.topk(topk, dim = 1)
      top_p = top_p.cpu().numpy()[0]
      top_class = top_class.cpu().numpy()[0]
      class_out = [inverted_class[k] for k in  top_class]
    return top_p, class_out

#make prediction 

top_proba, top_class = predict(Image, model, args.topk)
# return label

print(" Predicted class :" ,top_class)
print("Predicted proba : " , top_proba)