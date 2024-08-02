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
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

LR = 0.001
BATCH_SIZE = 4
EPOCHS = 15

transforms = A.Compose([
    A.Resize(height=412, width=274),
    A.Rotate(limit=60, p=0.85),
    A.HorizontalFlip(0.5),
    A.VerticalFlip(0.25),
    ToTensorV2()
])

def train(model, loss_fn, optimizer, train_dataloader, val_dataloader):
    train_loss, train_acc, val_loss, val_acc = 0, 0, 0, 0
    for batch, (X,y) in enumerate(tqdm(train_dataloader)):
        X,y = X.to(device), y.long().to(device)
        y = torch.squeeze(y, dim=1)
        model.train()

        pred = model(X)

        loss = loss_fn(pred, y)
        train_loss += loss.item()

        pred_labels = torch.argmax(pred, dim=1)
        y_flat = y.view(-1).cpu().numpy()
        pred_flat = pred_labels.view(-1).cpu().numpy()
        train_acc += accuracy_score(y_flat, pred_flat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()
        del X,y,pred,loss

    with torch.inference_mode():
      model.eval()
      for batch, (X,y) in enumerate(tqdm(val_dataloader)):
        X,y = X.to(device), y.long().to(device)

        pred = model(X)
        loss = loss_fn(pred, y)
        val_loss += loss.item()

        pred_labels = torch.argmax(pred, dim=1)
        y_flat = y.view(-1).cpu().numpy()
        pred_flat = pred_labels.view(-1).cpu().numpy()
        val_acc += accuracy_score(y_flat, pred_flat)

    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    val_loss /= len(val_dataloader)
    val_acc /= len(val_dataloader)
    print(f'Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f} | Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.4f}')

def test(model, loss_fn, optimizer, test_dataloader):
  test_loss, test_acc = 0,0
  with torch.inference_mode():
    model.eval()
    for batch, (X,y) in enumerate(tqdm(test_dataloader)):
      X,y = X.to(device), y.long().to(device)

      pred = model(X)
      loss = loss_fn(pred, y)
      test_loss += loss.item()

      pred_labels = torch.argmax(pred, dim=1)
      y_flat = y.view(-1).cpu().numpy()
      pred_flat = pred_labels.view(-1).cpu().numpy()
      test_acc += accuracy_score(y_flat, pred_flat)
  test_loss /= len(test_dataloader)
  test_acc /= len(test_dataloader)
  print(f'Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}')

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
    test_dl = DataLoader(test_dataset, batch_size=BATCH_SIZE, pin_memory=True, num_workers=1)
    
    for epoch in range(EPOCHS):
        train(model, loss_fn, optimizer, train_dl, val_dl)
    test(model, loss_fn, optimizer, test_dl)

    print("Saving the model...")
    torch.save(model.state_dict(), 'trained_unet_model.pth')
    print("Model saved successfully.")