import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim 
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

#SET THE SEED
torch.manual_seed(42)
torch.cuda.manual_seed(42)
random.seed(42)

#SET UP HYPERPARAMETERS
BATCH_SIZE=128
EPOCHS = 10
LEARNING_RATE = 3e-4
PATCH_SIZE=4
NUM_CLASSES=10
IMAGE_SIZE=32
CHANNELS = 3
EMBED_DIM = 256
NUM_HEADS = 8
DEPTH = 6 
MLP_DIM = 512
DROP_RATE = 0.1

#image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5),(0.5))
    #helps the model to convert faster
    #helps to make the numerical computations stable (normalize)                                
])

