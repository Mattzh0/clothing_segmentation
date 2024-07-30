import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

image_directory = r'clothing_data\jpeg_images\IMAGES'
mask_directory = r'clothing_data\jpeg_masks\MASKS'
image_list = os.listdir(image_directory)
mask_list = os.listdir(mask_directory)

image_list.sort()
mask_list.sort()

# Train (90%), Test (5%), Validation (5%)
train_img_list, temp_img_list, train_mask_list, temp_mask_list = train_test_split(image_list, mask_list, test_size=0.10, random_state=42)
val_img_list, test_img_list, val_mask_list, test_mask_list = train_test_split(temp_img_list, temp_mask_list, test_size=0.50, random_state=42)

class SegmentationDataset(Dataset):
    def __init__(self, img_dir, img_list, mask_dir, mask_list, transformations):
        self.img_dir = img_dir
        self.img_list = img_list
        self.mask_dir = mask_dir
        self.mask_list = mask_list
        self.transformations = transformations

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img = np.array(Image.open(os.path.join(self.img_dir, self.img_list[index])).convert('RGB'))
        mask = np.array(Image.open(os.path.join(self.mask_dir, self.mask_list[index])).convert('L'))
        mask = np.clip(mask, 0, 58)

        if self.transformations:
            transformed = self.transformations(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']
        return img, mask

