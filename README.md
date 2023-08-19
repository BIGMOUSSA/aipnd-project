# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

## installation and setup

create a environnement

python -m venv venv

install requirement

pip -r intall requirements.txt

## importants :
the dataset folder should have
 - train : folder
 - test : foldder
 - valid : folder

### run train.py for building a image classification model

python train.py [path_to_dataset] --save_dir --num_epoch --arch 

for example :

python train.py flowers --arch vgg13 --num_epoch 1 --save_dir checkpoints


## run predict.py for prediction

python predict.py [path_to_image] [path_to_checkpoint] --category_name cat_to_name.json --top_k 3 

option that begin with  "--" are optionals


