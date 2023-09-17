import cv2
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
import imagesize
import filetype


class SRDataset(Dataset):
    def __init__(self, hr_folder, lr_folder='lr', scale=2, update=True):
        self.hr_folder = hr_folder
        self.lr_folder = lr_folder
        self.scale = scale
        self.update = update

        self.names = self.read_folders()

    def __len__(self):
        return len(self.names)

    def read_folders(self):
        hr_names = sorted(os.listdir(self.hr_folder))
        hr_names = list(filter(lambda x: filetype.is_image(os.path.join(self.hr_folder, x)), hr_names))
        self.create_lr_images(hr_names)
        return hr_names

    def create_lr_images(self, hr_names):
        def create_lr_image(hr_path, lr_path):
            img = cv2.imread(hr_path)
            img = cv2.resize(img, None, fx=1 / self.scale, fy=1 / self.scale, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(lr_path, img)

        Path(self.lr_folder).mkdir(parents=True, exist_ok=True)
        for name in hr_names:
            lr_path = os.path.join(self.lr_folder, name)
            hr_path = os.path.join(self.hr_folder, name)
            if not os.path.exists(lr_path):
                create_lr_image(hr_path, lr_path)
            else:
                width_hr, height_hr = imagesize.get(hr_path)
                width_lr, height_lr = imagesize.get(lr_path)
                if (width_hr / self.scale != width_lr) or (height_hr / self.scale != height_lr):
                    create_lr_image(hr_path, lr_path)

    @staticmethod
    def check_names(hr_names, lr_names):
        res = True
        for i in range(len(hr_names)):
            if hr_names[i] != lr_names[i]:
                res = False
                break
        return res

    @staticmethod
    def load_image(path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, i):
        name = self.names[i]

        hr_path = os.path.join(self.hr_folder, name)
        lr_path = os.path.join(self.lr_folder, name)

        hr_img = self.load_image(hr_path)
        lr_img = self.load_image(lr_path)

        hr_t = to_tensor(hr_img)
        lr_t = to_tensor(lr_img)

        return hr_t, lr_t, name


def to_array(tensor):
    tensor = tensor.squeeze(0).clamp(0, 1)
    tensor = torch.permute(tensor, (1, 2, 0))
    tensor *= 255
    img = np.array(tensor, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img
