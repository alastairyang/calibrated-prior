import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import pathlib
import datetime

class Autoencoder(nn.Module):

    def __init__(self, image_dimension, batch_size, kernel_1_size = 3, kernel_2_size = 7):
      super(Autoencoder, self).__init__()

      dimension_after_conv = image_dimension - (kernel_1_size // 2) * 2 - (kernel_2_size // 2) * 2

      self.encoder = nn.Sequential(
        nn.Conv2d(1, 1, kernel_1_size, 1),
        nn.ReLU(),
        nn.Conv2d(1, 1, kernel_2_size, 1),
        nn.ReLU(),
        nn.Flatten(start_dim=1),
        nn.Linear(dimension_after_conv ** 2, 128),
        nn.Linear(128, 32),
        nn.Sigmoid()
      )
      self.decoder = nn.Sequential(
        nn.Sigmoid(),
        nn.Linear(32, 128),
        nn.Linear(128, dimension_after_conv ** 2),
        nn.Unflatten(dim=1, unflattened_size = (1, dimension_after_conv, dimension_after_conv)),
        nn.ReLU(),
        nn.ConvTranspose2d(1, 1, kernel_2_size, 1),
        nn.ReLU(),
        nn.ConvTranspose2d(1, 1, kernel_1_size, 1)
      )
    
    def main(self, x):
       x = self.encoder(x)
       x = self.decoder(x)
       return x