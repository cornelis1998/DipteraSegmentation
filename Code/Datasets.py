import glob
import os
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation

class MAE_Dataset(Dataset):
    def __init__(self, data_folders, num_channels=1, do_transforms=True, mask_ratio=0.25, patch_size=8):
        assert 64 % patch_size == 0, "Patch size must be a factor of 64"

        self.num_channels = num_channels
        self.data_folder = data_folders
        self.file_names = []
        for folder in data_folders:
            for root, dirs, _ in os.walk(folder):
                for dir in dirs:
                    self.file_names.extend(glob.glob(os.path.join(root, dir, "*.npz")))


        print(f"Found {len(self.file_names)} files")

        self.transforms = Compose([
            RandomHorizontalFlip(p=0.5),  # Apply horizontal flip to 50% of images
            RandomVerticalFlip(p=0.5),  # Apply vertical flip to 50% of images
            RandomRotation([90, 90]),  # Rotate by 90 degrees
        ])
        self.do_transforms = do_transforms
        self.mask_ratio = mask_ratio  # Ratio of the image to mask
        self.patch_size = patch_size  # Size of the patches to be masked

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = self.file_names[idx]
        image = np.load(file_path)["arr_0"]

        img_Tensor = torch.tensor(image).unsqueeze(0).float()

        # Normalize to range [0, 1]
        min = torch.min(img_Tensor)
        max = torch.max(img_Tensor)
        img_Tensor = (img_Tensor - min) / (max - min)

        if self.do_transforms:  # Apply the transformation
            img_Tensor = self.transforms(img_Tensor)

        # Create masked version of the image
        if self.mask_ratio > 0.01:
            masked_img_Tensor = apply_mask(img_Tensor.clone(), mask_ratio=self.mask_ratio)
        else:
            masked_img_Tensor = img_Tensor.clone()

        # repeat the image to have 3 channels
        # img_Tensor = img_Tensor.repeat(self.num_channels, 1, 1)
        masked_img_Tensor = masked_img_Tensor.repeat(self.num_channels, 1, 1)

        return masked_img_Tensor, img_Tensor  # Return both masked input and original target

class MUnet_Dataset(Dataset):
    def __init__(self, data_folder, num_channels=1, mask_ratio=0.25, do_transforms=True, patch_size=8):
        self.num_channels = num_channels
        self.data_folder = data_folder
        self.file_names = [f for f in os.listdir(data_folder) if f.endswith(".npz")]

        self.transforms = Compose([
            RandomHorizontalFlip(p=0.5),  # Apply horizontal flip to 50% of images
            RandomVerticalFlip(p=0.5),  # Apply vertical flip to 50% of images
            RandomRotation([90, 90]),  # Rotate by 90 degrees
        ])
        self.do_transforms = do_transforms
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_folder, self.file_names[idx])
        data = np.load(file_path)

        img = data['frame'].astype(np.float32)
        mask = data['mask'].astype(np.float32)
        weight = data['weight'].astype(np.float32)

        # Normalize the image
        img = img - img.min()
        img = img / img.max()

        # Stack img and mask to ensure the same transformation
        img_mask = np.stack([img, mask], axis=0)  # Stack along new axis to keep channels separate
        img_mask = torch.tensor(img_mask)

        if self.do_transforms:  # Apply the transformation
            img_mask = self.transforms(img_mask)

        # print(f"input min: {img_mask.min()}, max: {img_mask.max()}")

        # Split them back into separate tensors
        img = img_mask[0].unsqueeze(0)  # Maintain the channel dimension for img
        mask = img_mask[1].unsqueeze(0)  # Maintain the channel dimension for mask

        # make the chance 50% to apply the mask
        if self.mask_ratio > 0.01:
            img = apply_mask(img, mask_ratio=self.mask_ratio, patch_size=self.patch_size)

        img = img.repeat(self.num_channels, 1, 1)

        return img, mask, weight

def apply_mask(img_Tensor, patch_size=8, mask_ratio=0.25):
    _, H, W = img_Tensor.shape
    num_patches = (H // patch_size) * (W // patch_size)
    num_masked_patches = int(num_patches * mask_ratio)

    # Create a mask with ones
    mask = torch.ones_like(img_Tensor)

    # Randomly select patches to mask
    patch_indices = torch.randperm(num_patches)[:num_masked_patches]

    for idx in patch_indices:
        row = (idx // (W // patch_size)) * patch_size
        col = (idx % (W // patch_size)) * patch_size
        mask[:, row:row + patch_size, col:col + patch_size] = 0

    masked_img_Tensor = img_Tensor * mask

    return masked_img_Tensor


def calculate_weights(labels, sigma=5):
    # Assuming labels is a binary mask with cells labeled as 1 and background as 0
    # Compute distance to the nearest cell (d1) and second nearest cell (d2)
    d1 = distance_transform_edt(labels)
    d2 = distance_transform_edt(1 - labels)

    # Compute the weight map
    wc = 1  # Placeholder, needs proper calculation based on class frequencies
    w0 = 10
    w = wc + w0 * np.exp(- (d1 + d2) ** 2 / (2 * sigma ** 2))

    return torch.tensor(w, dtype=torch.float32)


