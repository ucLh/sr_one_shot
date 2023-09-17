import os
from pathlib import Path
from typing import List

import cv2
import filetype
import imagesize
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


class SRDataset(Dataset):
    """
    Simple dataset class. Can produce low resolution images given high resolution ones
    """
    def __init__(self, hr_folder: str, lr_folder: str = 'lr', scale: int = 4):
        self.hr_folder = hr_folder
        self.lr_folder = lr_folder
        assert os.path.isdir(hr_folder), '`hr_folder` should be a directory'
        # This class expects low resolution images to be `scale` times smaller than the high resolution ones.
        # If that is false, the class will produce and save low res images itself
        self.scale = scale

        self.names = self.read_folders()

    def __len__(self) -> int:
        return len(self.names)

    def read_folders(self) -> List[str]:
        # Sort images for consistency
        hr_names = sorted(os.listdir(self.hr_folder))
        # Filter out non-image filenames
        hr_names = list(filter(lambda x: filetype.is_image(os.path.join(self.hr_folder, x)), hr_names))
        # Creates lr images if needed
        self.create_lr_images(hr_names)
        return hr_names

    def create_lr_images(self, hr_names: List[str]):
        def create_lr_image(hr_path, lr_path):
            # Downsize the image and save it
            img = cv2.imread(hr_path)
            img = cv2.resize(img, None, fx=1 / self.scale, fy=1 / self.scale, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(lr_path, img)

        Path(self.lr_folder).mkdir(parents=True, exist_ok=True)
        for name in hr_names:
            lr_path = os.path.join(self.lr_folder, name)
            hr_path = os.path.join(self.hr_folder, name)
            # Create image if it doesn't exist or it's size is not equal to (high res image size) / scale
            if not os.path.exists(lr_path):
                create_lr_image(hr_path, lr_path)
            else:
                width_hr, height_hr = imagesize.get(hr_path)
                width_lr, height_lr = imagesize.get(lr_path)
                if (width_hr / self.scale != width_lr) or (height_hr / self.scale != height_lr):
                    create_lr_image(hr_path, lr_path)

    @staticmethod
    def load_image(path: str) -> np.ndarray:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, i: int) -> (torch.Tensor, torch.Tensor, str):
        name = self.names[i]

        hr_path = os.path.join(self.hr_folder, name)
        lr_path = os.path.join(self.lr_folder, name)

        hr_img = self.load_image(hr_path)
        lr_img = self.load_image(lr_path)

        hr_t = to_tensor(hr_img)
        lr_t = to_tensor(lr_img)

        return hr_t, lr_t, name


def to_array(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert network output back to array in order to save the image
    """
    tensor = tensor.squeeze(0).clamp(0, 1)
    tensor = torch.permute(tensor, (1, 2, 0))
    tensor *= 255
    img = np.array(tensor, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img
