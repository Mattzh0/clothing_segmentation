from model import UNET_Model
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

LR = 0.001
BATCH_SIZE = 32
EPOCHS = 20

transforms = A.Compose([
    A.Resize(height=824, width=548)
])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

