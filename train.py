import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
from collections import OrderedDict

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable

import helpers.DataLoading
import argparse

# Creates Argument Parser object named parser
# Set directory to save checkpoints: 
# python train.py data_dir --save_dir save_directory
args_parser = argparse.ArgumentParser(description="Sets arguments for neural network construction and training")
args_parser.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
args_parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
# Choose architecture: 
# python train.py data_dir --arch "vgg13"
args_parser.add_argument('--arch', dest="arch", action="store", default="vgg19", type = str )
# Set hyperparameters: 
# python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
args_parser.add_argument('--learning_rate', dest="Learning Rate", action="store", default="0.01")
args_parser.add_argument('--hidden_units', dest="Hidden units", action="store", type=int, default=512)
args_parser.add_argument('--epochs', dest="Epochs", action="store", type=int, default=7)
# Use GPU for training: \
# python train.py data_dir --gpu
args_parser.add_subparsers('--gpu', dest="gpu", action="store", default="gpu")

args = args_parser.parse_args()
data_dir = args.data_dir 
save_dir = args.save_dir
arch = args.arch
learning_rate = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
gpu_or_cpu = args.gpu
#dropout = args.dropout

data_loaders = helpers.DataLoading.load_image_and_data(data_dir)
model, criterion, optimizer = helpers.NeuralNetUtil.create_model(arch, hidden_units,learning_rate)
helpers.NeuralNetUtils.train_network(model, optimizer, criterion, epochs, 20, data_loaders['training'], gpu_or_cpu)
helpers.NeuralNetUtils.save_model(model, datasets, learning_rate, epochs, criterion, optimizer, hidden_units, arch, 32)

print("Model trained") 