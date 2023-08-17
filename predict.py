
# import package
import argparse
import json
import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
from utils import load_checkpoint, process_image, print_result, plot_result
from models_utils import predict


# definir arg parameters

parser = argparse.ArgumentParser(description = "Application to predict a image label")
parser.add_argument('image_path', help = 'the path to the image to predict')
parser.add_argument('checkpoint', help = 'path to the checkpoint model', default= "checkpoints/checkpoint.pt")
parser.add_argument('--top_k', help = "the number of top classe after prediction", type = int, default = 1)
parser.add_argument('--category_name', help = " a json file for the category name", default= "cat_to_name.json")
parser.add_argument('--gpu', default = "gpu")

args = parser.parse_args()

model, optimizer, num_epoch = load_checkpoint(args.checkpoint)
# lead file and make preprocessing
image = process_image(args.image_path)


#predict label
if args.gpu == "gpu" :
    # Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device('cpu')   
#make prediction 

top_proba, top_class = predict(args.image_path, model, args.top_k, device)
# return label / name

with open(args.category_name, 'r') as f:
    cat_to_name = json.load(f)

predicted_flowers = [cat_to_name[idx] for idx in top_class]
#print(" Predicted class :", predicted_flowers)
#print("Predicted proba : " , top_proba)
top_proba = [float(p) for p in top_proba]
print_result(predicted_flowers, top_proba)

plot_result(im_path=args.image_path, top_proba=top_proba, predicted=predicted_flowers, topk=args.top_k)