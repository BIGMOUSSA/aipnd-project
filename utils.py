# Imports here
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np

def save_checkpoint(model, checkpoint_dir):
    checkpoint_filename = f'checkpoint_{epoch}_{avg_valid_accuracy:.4f}_{current_datetime}.pt'
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

    state = {
        'input_size' : 25088,
        'output_size' : 102,
        'classifier': model.classifier,
        'num_epoch' : epochs,
        'optimizer' : optimizer.state_dict(),
        'model_state_dict': model.state_dict(),
        'class_to_idx' : model.class_to_idx
    }

    torch.save(state, checkpoint_path)

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    # Create a new model instance with the same architecture as saved in the checkpoint
    model = vgg16()
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