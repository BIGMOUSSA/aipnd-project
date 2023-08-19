# Imports here
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import json

def save_checkpoint(model, checkpoint_dir, optimizer, epochs):
    '''
        save the checkpoint after training
    '''
    checkpoint_filename = f'checkpoint.pt'
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

    state = {
        'input_size' : 25088,
        'output_size' : 102,
        'classifier': model.classifier,
        'num_epoch' : epochs,
        'optimizer' : optimizer.state_dict(),
        'model_state_dict': model.state_dict(),
        'class_to_idx' : model.class_to_idx,
        'model_arch' : model
    }

    torch.save(state, checkpoint_path)

def load_checkpoint(checkpoint_path):
    '''
        Load the checkpoint for prediction
        args : checkpoint path
    '''
    checkpoint = torch.load(checkpoint_path)

    # Create a new model instance with the same architecture as saved in the checkpoint
    model = checkpoint["model_arch"]
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    num_epoch = checkpoint['num_epoch']

    # You might need to re-create the optimizer with the same parameters as used during training
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
    optimizer.load_state_dict(checkpoint['optimizer'])

    # Retrieve additional information from the checkpoint


    return model, optimizer, num_epoch


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)
    im.thumbnail([256, 256])
    # Crop the center 224x224 portion of the image
    left = (im.width - 224) / 2
    top = (im.height - 224) / 2
    right = left + 224
    bottom = top + 224
    im = im.crop((left, top, right, bottom))
    # im to array and normalize
    np_image = np.array(im)/255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    # reshaping to fit tensor
    np_image = np_image.transpose(2,1,0)
    return torch.tensor(np_image).unsqueeze(0).float()


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image[0].numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

def check_dataset_folders(dataset_path):
    train_folder = os.path.join(dataset_path, 'train')
    test_folder = os.path.join(dataset_path, 'test')
    valid_folder = os.path.join(dataset_path, 'valid')
    
    if not os.path.exists(train_folder):
        raise FileNotFoundError(f"Train folder '{train_folder}' not found.")
    if not os.path.exists(test_folder):
        raise FileNotFoundError(f"Test folder '{test_folder}' not found.")
    if not os.path.exists(valid_folder):
        raise FileNotFoundError(f"Valid folder '{valid_folder}' not found.")

    print("All required dataset folders found.")

def print_result(top_class, top_proba) :
    mydict = {}
    
    for k,v in zip(top_class, top_proba):
        mydict[k] = v
    
    print(json.dumps(mydict, indent=2))

def plot_result(im_path, predicted, top_proba, topk):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    image = Image.open(im_path)
    plt.imshow(image)
    plt.axis('off')
    plt.title('Input Image')
    
    plt.subplot(1, 2, 2)
    y_pos = np.arange(len(predicted))
    plt.barh(y_pos, top_proba, align='center')
    plt.yticks(y_pos, predicted)
    plt.gca().invert_yaxis()  # Invert y-axis to show the highest probability at the top
    plt.xlabel('Probability')
    plt.title('Top ' + str(topk) + ' Predicted Classes')
    plt.tight_layout()
    
    plt.show()