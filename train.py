import model
import dataset
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

LR = 0.001
BATCH_SIZE = 4
EPOCHS = 3

transforms = A.Compose([
    A.Resize(height=824, width=548),
    A.Rotate(limit=360, p=0.85),
    A.Normalize(max_pixel_value=255.0),
    A.HorizontalFlip(0.5),
    A.VerticalFlip(0.25),
    ToTensorV2()
])

def train(model, loss_fn, optimizer, data_loader):
    for batch, (X,y) in enumerate(tqdm(data_loader)):
        X,y = X.to(device), y.long().squeeze(1).to(device)
        y = torch.squeeze(y, dim=1)
        model.train()

        pred = model(X)

        print(f"Prediction Shape: {pred.shape} | Target Shape: {y.shape}")

        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    model = model.UNET_Model().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    img_dir = dataset.image_directory
    mask_dir = dataset.mask_directory

    train_dataset = dataset.SegmentationDataset(img_dir, dataset.train_img_list, mask_dir, dataset.train_mask_list, transformations=transforms)
    val_dataset = dataset.SegmentationDataset(img_dir, dataset.val_img_list, mask_dir, dataset.val_mask_list, transformations=transforms)
    test_dataset = dataset.SegmentationDataset(img_dir, dataset.test_img_list, mask_dir, dataset.test_mask_list, transformations=transforms)

    train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True)
    val_dl = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True)
    
    for epoch in range(EPOCHS):
        train(model, loss_fn, optimizer, train_dl)