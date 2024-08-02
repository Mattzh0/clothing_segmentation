
import dataset
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

transforms = A.Compose([
    A.Resize(height=416, width=288),
    ToTensorV2()
])

model_path = r'models\trained_deeplabv3plus_model.pth'
model = smp.DeepLabV3Plus(encoder_name="resnet101", encoder_weights="imagenet", classes=59)
model.load_state_dict(torch.load(model_path))

img_dir = dataset.image_directory
mask_dir = dataset.mask_directory

test_dataset = dataset.SegmentationDataset(img_dir, dataset.test_img_list, mask_dir, dataset.test_mask_list, transformations=transforms)

def get_test_sample(test_dataset, index):
    # get a sample from the test dataset
    sample_image, sample_mask = test_dataset[index]

    # add the batch dimension to the sample image
    sample_image_batch = sample_image.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        predicted_mask = model(sample_image_batch)
    predicted_mask = torch.argmax(predicted_mask, dim=1)

    # remove the batch dimension and move the tensors to CPU
    predicted_mask = predicted_mask.squeeze(0).cpu()
    sample_image = sample_image.cpu()
    sample_mask = sample_mask.cpu()

    return sample_image, sample_mask, predicted_mask

def plot_images(sample_image, sample_mask, predicted_mask):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(sample_image.permute(1, 2, 0))
    ax[0].set_title('Original Image')

    ax[1].imshow(sample_mask, cmap='gray')
    ax[1].set_title('Ground Truth Mask')

    ax[2].imshow(predicted_mask, cmap='gray')
    ax[2].set_title('Predicted Mask')

    plt.show()

def generate_shades(predicted_mask):
    img_array = predicted_mask.numpy()
    shades = np.unique(img_array)

    shaded_images = []

    output_dir = 'shades'
    os.makedirs(output_dir, exist_ok=True)
    for shade in shades:
        # create initial fully transaprent image
        shaded_image = Image.new('RGBA', img_array.shape[::-1], (0, 0, 0, 0))
        shaded_array = np.array(shaded_image)

        # create a mask where pixels match the current shade
        mask = (img_array == shade)
        shaded_array[mask] = (shade, shade, shade, 255)  # (R, G, B, A)

        # convert array back to image and append to list
        shaded_images.append(Image.fromarray(shaded_array, 'RGBA'))

    # save each shaded image
    for i, img in enumerate(shaded_images):
        img_path = os.path.join(output_dir, f'shade_{i}.png')
        img.save(img_path)

def invert_mask_zero():
    mask_zero_path = r'shades\shade_0.png'
    img = Image.open(mask_zero_path)
    img_array = np.array(Image.open(mask_zero_path))

    # create image with the same dimensions as the mask
    height, width, _ = img_array.shape
    inverted_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    inverted_array = np.array(inverted_image)

    # invert the mask (previously transparent becomes black, previously black becomes transparent)
    # img_array[:, :, 3] refers to the alpha channel (transparency)
    inverted_array[img_array[:, :, 3] == 0] = (0, 0, 0, 255)
    inverted_array[img_array[:, :, 3] == 255] = (0, 0, 0, 0)

    # convert the array back to an image and save it
    inverted_image = Image.fromarray(inverted_array, 'RGBA')

    img_path = os.path.join('inverted_mask', 'inverted_mask_zero.png')
    inverted_image.save(img_path)

def apply_original_colors():
    pass

if __name__ == '__main__':
    sample_image, sample_mask, predicted_mask = get_test_sample(test_dataset, 3)
    generate_shades(predicted_mask)
    plot_images(sample_image, sample_mask, predicted_mask)
    invert_mask_zero()