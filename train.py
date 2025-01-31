import argparse
import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
from utils import check_dataset_folders, save_checkpoint
from models_utils import EarlyStopping

parser = argparse.ArgumentParser(description = "Application that train a deeplearnig model classification for a given dataset")
parser.add_argument('dataset', help = 'the path to the dataset folder')
parser.add_argument('--save_dir', help = 'the path to the folder for saving checkpoint', default= "checkpoints")
parser.add_argument('--arch', help = "specify which pretrained checkpoint to use for finetuning", default= "vgg13", choices = ["vgg11", "vgg13", "vgg16"])
parser.add_argument('--gpu', help = 'tell weither to use gpu or cpu', default = 'gpu', choices = ['cpu', 'gpu'])
parser.add_argument('--learning_rate', help = "specify the learning rate for the model learning", default = 0.003, type = float)
parser.add_argument('--num_epoch', help = 'give the number of epoch for the training loop', default = 5, type = int)
parser.add_argument('--hidden_unit', help = 'give the hidden laye size', default = 4096, type = int)


args = parser.parse_args()

## Let's check if the dataset is well organized

# Check if the required folders exist
check_dataset_folders(args.dataset)

# check if the saving directory exist

if not os.path.exists(args.save_dir) :
    raise FileNotFoundError(f"folder for saving model '{args.save_dir}' not found.")

# i take the data and preprocess them
data_dir = args.dataset
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

### Data processing
# TODO: Define your transforms for the training, validation, and testing sets

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
# TODO: Load the datasets with ImageFolder
train_datasets =  datasets.ImageFolder(train_dir, transform = train_transforms)
test_datasets =  datasets.ImageFolder(test_dir, transform = test_transforms)
valid_datasets =  datasets.ImageFolder(valid_dir, transform = test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_dataloaders = torch.utils.data.DataLoader(train_datasets, batch_size = 64, shuffle = True)
test_dataloaders = torch.utils.data.DataLoader(test_datasets, batch_size = 64, shuffle = True)
valid_dataloaders = torch.utils.data.DataLoader(valid_datasets, batch_size = 64, shuffle = True)

# let's set the number of class
# List all subdirectories (classes) within the dataset directory
class_names = os.listdir(train_dir)

# Count the number of classes
num_classes = len(class_names)
# i download the torchvision model given as checkpoint

if args.arch == "vgg13" : 
    if os.path.exists('vgg13-19584684.pth'):
        model = models.vgg13(weights=None)
        checkpoint_path = 'vgg13-19584684.pth'
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path)
        # Load the weights into the model
        model.load_state_dict(checkpoint)
    else :
        model = models.vgg13(weights="DEFAULT")
elif args.arch == "vgg11" :
    model = models.vgg11(weights="DEFAULT")
elif args.arch == "vgg16" :
    model = models.vgg16(weights="DEFAULT")
else :
    print("Not yet handled by the application")
model

if args.gpu == "gpu" :
    # Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device('cpu')

# lets freeze the parameter
for param in model.parameters():
    param.requires_grad = False

# i redefine my model architecture
model.classifier = nn.Sequential(nn.Linear(25088, args.hidden_unit),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(args.hidden_unit, num_classes),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()

# i set up optimizer and learning rate
optimizer = optim.Adam(model.classifier.parameters(), lr = args.learning_rate)
model.to(device);

# Instantiate the EarlyStopping object

early_stopping = EarlyStopping(patience = 25, delta = 0.001)

# i write the training loop and testing

epochs = args.num_epoch
steps = 0
print_every = 20

train_losses = []
test_losses = []
test_accuracies = []
print("The training loop is started ...")
for epoch in range(epochs):
    running_loss = 0
    model.train()
    for inputs, labels in train_dataloaders:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)

        logps = model.forward(inputs)
        loss = criterion(logps, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            model.eval()
            test_loss = 0
            accuracy = 0
            with torch.no_grad():
                for test_inputs, test_labels in test_dataloaders:
                    test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
                    test_logps = model.forward(test_inputs)
                    batch_loss = criterion(test_logps, test_labels)

                    test_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(test_logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == test_labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            train_losses.append(running_loss / print_every)
            test_losses.append(test_loss / len(test_dataloaders))
            test_accuracies.append(accuracy / len(test_dataloaders))

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {train_losses[-1]:.3f}.. "
                  f"Test loss: {test_losses[-1]:.3f}.. "
                  f"Test accuracy: {test_accuracies[-1]:.3f}")
            
            # Check if early stopping criteria met
            early_stopping(test_losses[-1])

            if early_stopping.early_stop:
                print("Early stopping")
                break

            running_loss = 0
            model.train()
print(" Training loop finished .")
# printing evaluation score
valid_losses = []
valid_accuracies = []

model.eval()
with torch.no_grad():
    for inputs, labels in valid_dataloaders:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model(inputs)
        batch_loss = criterion(logps, labels)

        # Accumulate test loss and accuracy

        valid_losses.append(batch_loss.item())
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy = torch.mean(equals.type(torch.FloatTensor))
        valid_accuracies.append(accuracy.item())

# Calculate mean loss and accuracy over the validation set

avg_valid_loss = sum(valid_losses) / len(valid_losses)
avg_valid_accuracy = sum(valid_accuracies) / len(valid_accuracies)

# after training print the validation score
print(f"Validation Loss: {avg_valid_loss:.3f}, Validation Accuracy: {avg_valid_accuracy:.3f}")

model.class_to_idx = train_datasets.class_to_idx


checkpoint_folder = args.save_dir
os.makedirs(checkpoint_folder, exist_ok=True)

save_checkpoint(model, checkpoint_folder,  optimizer = optimizer, epochs = epochs)