import csv
import os
import random
from typing import Any, Dict, List, Tuple
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision import transforms


class FoodDatasetWithSE(Dataset):
    """
    Dataset containing SingleFood and AIFood
    
    Arguments:
        root `str`: Path to csv file containing paths and labels of all image files. \
        CSV file is seperated by comma, the first item is path to the image, the rests are labels in multi-hot format.
        Ex (an image with label 2 and 4): `./path/to/image.jpg,0,0,1,0,1`
        transform `Transform`: Transformation to be apply on images. If not specified, \
        a default transform is applied to convert PIL Image to Tensor.
    """

    def __init__(
        self,
        dataset_root: str,
        se_root: str,
        img_csv: str,
        random_crop_size: Tuple[int, int],
        random_resize_long_side_range: Tuple[int, int] = None,
        color_jitter_brightness: float = None,
        color_jitter_contrast: float = None,
        color_jitter_saturation: float = None,
        color_jitter_hue: float = None,
    ):
        self.dataset_root = dataset_root
        self.se_root = se_root
        self.datalist = self._get_datalist(img_csv)

        self.random_crop_size = random_crop_size
        self.random_resize_long_side_range = random_resize_long_side_range
        self.color_jitter_transform = transforms.ColorJitter(
            brightness=color_jitter_brightness,
            contrast=color_jitter_contrast,
            saturation=color_jitter_saturation,
            hue=color_jitter_hue,
        )

    def _get_datalist(self, img_csv: str) -> List[Tuple[str, List[int]]]:
        datalist = []
        with open(img_csv, "r") as file:
            csv_reader = csv.reader(file, delimiter=",")
            # Each label is in multi-hot form
            for img_rel_path, *label in csv_reader:
                datalist.append((img_rel_path, list(int(i) for i in label)))
        return datalist

    def _transform(self, img: Image.Image, se: np.ndarray):
        """
        Apply transformations on image and segmentation map
        Args:
            img (Image.Image): Image to be transformed
            se (np.ndarray): Segmentation map to be transformed
        Returns:
            img (torch.Tensor): Transformed image with shape (3, H, W)
            se (torch.Tensor): Transformed segmentation map with shape (1, H, W)
        """
        # se Image
        se = (se + 1).astype(np.uint8)  # [0, segmentation_objects]
        se = np.expand_dims(se, axis=2)
        se = np.repeat(se, 3, axis=2)
        print(np.unique(se))
        se = Image.fromarray(se)

        # random resize
        target_long_side = random.randint(
            self.random_resize_long_side_range[0],
            self.random_resize_long_side_range[1],
        )
        w, h = img.size
        if w < h:
            target_shape = (int(round(w * target_long_side / h)), target_long_side)
        else:
            target_shape = (target_long_side, int(round(h * target_long_side / w)))
        img = img.resize(target_shape, resample=Image.CUBIC)
        se = se.resize(target_shape, resample=Image.NEAREST)

        # random horizontal flip
        if random.random() < 0.5:
            img = F.hflip(img)
            se = F.hflip(se)

        # color jitter
        if random.random() < 0.8:
            img = self.color_jitter_transform(img)

        # random grayscale
        if random.random() < 0.02:
            img = F.to_grayscale(img, num_output_channels=3)

        # random crop
        if self.random_crop_size is not None:
            crop_size = self.random_crop_size[0]
            # padding
            w, h = img.size
            pad_w = max(0, crop_size - w)
            pad_h = max(0, crop_size - h)
            padding = (
                pad_w // 2,
                pad_h // 2,
                pad_w - pad_w // 2,
                pad_h - pad_h // 2,
            )
            img = F.pad(img, padding, fill=0, padding_mode="constant")
            se = F.pad(se, padding, fill=0, padding_mode="constant")

            # crop
            i, j, h, w = transforms.RandomCrop.get_params(
                img, output_size=(crop_size, crop_size)
            )
            img = F.crop(img, i, j, h, w)
            se = F.crop(se, i, j, h, w)

        # img to tensor, scale to 0-1
        img = F.to_tensor(img)  # (C,H,W)

        # normalize
        img = F.normalize(
            img,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # reindex se
        se = np.array(se)
        unique_values = np.unique(se)
        sorted_values = np.sort(unique_values)
        mapped_img = np.searchsorted(sorted_values, se)
        # se to tensor
        se = torch.from_numpy(mapped_img).permute(2, 0, 1)  # (C,H,W)
        se = se[0].unsqueeze(0)  # (1,H,W)

        return img, se

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get an image from dataset

        Return:
            pack `Dict[str, Any]`: Dictionary containing image(Tensor), se(Tensor), one-hot label(Tensor) and name(str)
        """
        img_rel_path, label = self.datalist[index]
        img_path = os.path.join(self.dataset_root, img_rel_path)
        se_path = os.path.join(self.se_root, img_rel_path)
        se_path = os.path.splitext(se_path)[0] + ".npy"

        img = Image.open(img_path).convert("RGB")
        se = np.load(se_path)

        img, se = self._transform(img, se)

        pack = {
            "img": img,
            "se": se,
            "label": torch.tensor(label),
            "name": img_rel_path,
        }
        return pack

    def __len__(self):
        return len(self.datalist)
