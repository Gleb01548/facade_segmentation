import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from PIL import Image


class BuildDataset(Dataset):
    def __init__(
        self,
        folder,
        file_list,
        transform=None,
        transform_other=None,
        tr_chance=0.6,
        use_norm=True,
    ):
        self.transform = transform
        self.transform_other = transform_other
        self.norm = Normalize(
            mean=[123.4683, 116.7980, 107.3399], std=[4593.9688, 4234.8867, 3840.9556]
        )
        self.use_norm = use_norm

        self.path = folder
        self.file_list = file_list
        self.tr_chance = tr_chance

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        x_name = self.file_list[index]
        y_name = self.file_list[index].replace(".jpg", ".png")

        img_x = np.array(Image.open(f"{self.path}/{x_name}"), dtype="float32")
        img_y = np.array(Image.open(f"{self.path}/{y_name}"), dtype="float32")

        transformed = self.transform(image=img_x, mask=img_y)
        img_x = transformed["image"]
        img = self.transform(image=np.array(Image.open(f"{self.path}/{x_name}")))[
            "image"
        ]
        img_y = transformed["mask"]

        chance_num = random.uniform(0, 1)

        if self.transform != None and chance_num < self.tr_chance:
            transformed = self.transform_other(image=img_x, mask=img_y)
            img_x = transformed["image"]
            img_y = transformed["mask"]

        img_y[np.logical_and(img_y != 3, img_y != 8)] = 0
        img_y[np.logical_or(img_y == 3, img_y == 8)] = 1

        img_y = F.one_hot(torch.from_numpy(img_y).long())[:, :, 1]

        img_x = torch.from_numpy(img_x).permute(2, 0, 1)

        if self.use_norm:
            img_x = self.norm(img_x)

        return {"img": img, "img_x": img_x, "img_y": img_y, "x_name": x_name}