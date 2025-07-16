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

train_dataset = datasets.CIFAR10(root = "data",
                                 train = True,
                                 download = True,
                                 transform = transform
                                 
                                 )

test_dataset = datasets.CIFAR10(root = "data",
                                train = False,
                                download = True,
                                transform = transform
                                )

#convert dataset into dataloader
train_loader = DataLoader(dataset = train_dataset,
                          batch_size = BATCH_SIZE,
                          shuffle = True)

test_loader = DataLoader(dataset = test_dataset,
                         batch_size = BATCH_SIZE,
                         shuffle = False)

# check dataloader
print(f"Dataloader: {train_loader,test_loader}")
print(f"Length of train_loader: {len(train_loader)} batches of {BATCH_SIZE}")
print(f"Length of test_loader: {len(test_loader)} batches of {BATCH_SIZE}")


#the vision transformer
class PatchEmbedding(nn.Module):
  def __init__(self,img_size, patch_size,in_channels , embed_dim):
    super().__init__()
    self.patch_size = patch_size;
    self.proj = nn.Conv2d(in_channels = in_channels,
                          out_channels = embed_dim,
                          kernel_size = patch_size,
                          stride = patch_size
                          )
    num_patches = (img_size//patch_size)**2
    self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
    self.pos_embed = nn.Parameter(torch.randn(1,1+num_patches), embed_dim)


    def forward(self,x:torch.Tensor):
      B=x.size(0)
      x=self.proj(x) #(B,E,H/P,W/P)
      x=x.flatten(2).transpose(1,2) #(B,N,E)
      cls_token = self.cls_token(B,-1,-1)
      x = torch.cat((cls_token,x),dim=1)
      x=x+self.pos
      return x
    
def MLP(nn.Module):
  def __init__(self,
               in_feature,
               hidden_features.
               drop_rate):
    super().__init__()
    self.fc1 = nn.Linear(in_features = in_feature, out_features = hidden_features)
    self.fc2 = nn.Linear(in_features=hidden_features,
                         out_features = in_features)
    self.dropout = nn.Dropout(drop_rate)

  def forward(self,x):
    x=self.dropout(F.gelu(self.fc1(x)))
    x=self.dropout(self.fc2(x))
    return x
