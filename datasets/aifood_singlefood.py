import csv
import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from tools import imutils

IMG_FOLDER_NAME = "JPEGImages"


def load_img_name_list(dataset_path):
    img_gt_name_list = open(dataset_path).read().splitlines()
    img_name_list = [
        img_gt_name.split(" ")[0][-15:-4] for img_gt_name in img_gt_name_list
    ]

    return img_name_list


def get_img_path(img_name, voc12_root):
    return os.path.join(voc12_root, IMG_FOLDER_NAME, img_name + ".jpg")


def load_image_label_list_from_npy(img_name_list):
    cls_labels_dict = np.load("voc12/cls_labels.npy", allow_pickle=True).item()

    return [cls_labels_dict[img_name] for img_name in img_name_list]


class FoodDataset(Dataset):
    """
    Dataset containing SingleFood and AIFood
    
    Arguments:
        root `str`: Path to csv file containing paths and labels of all image files. \
        CSV file is seperated by comma, the first item is path to the image, the rests are labels in multi-hot format.
        Ex (an image with label 2 and 4): `./path/to/image.jpg,0,0,1,0,1`
        transform `Transform`: Transformation to be apply on images. If not specified, \
        a default transform is applied to convert PIL Image to Tensor.
    """

    def __init__(self, root: str, transform=None):

        self.transform = transform

        # List containing paths and labels of all images
        # Each label is in multi-hot form
        self.datalist = []
        with open(root, "r") as file:
            csv_reader = csv.reader(file, delimiter=",")
            for img_path, *label in csv_reader:
                self.datalist.append((img_path, list(int(i) for i in label)))

        # print(self.datalist)

    def get_image(self, img_path: str, transform=None) -> torch.Tensor:
        """
        Open an image and apply the transform
        """

        img = Image.open(img_path).convert("RGB")
        # Default transformation
        # Crop the image to 244x244 for ResNet50 input
        if transform is None:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop(224),
                ]
            )
        img = transform(img)
        return img

    def __getitem__(self, index: int):
        """
        Get an image from dataset

        Return:
            img `Tensor`: Image in Tensor format
            label `Tensor`: multi-hot label
        """

        img_path, label = self.datalist[index]
        img = self.get_image(img_path, self.transform)
        return img, torch.tensor(label)

    def __len__(self):
        return len(self.datalist)


class VOC12ImageDataset(Dataset):
    def __init__(self, img_name_list_path, voc12_root, transform=None):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root
        self.transform = transform

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]

        img = Image.open(get_img_path(name, self.voc12_root)).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return name, img


class VOC12ClsDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, voc12_root, transform=None):
        super().__init__(img_name_list_path, voc12_root, transform)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)

        label = torch.from_numpy(self.label_list[idx])

        pack = {}
        pack["img"] = img
        pack["label"] = label
        pack["name"] = name

        return pack


class VOC12ClsDatasetMSF(VOC12ClsDataset):
    def __init__(
        self,
        img_name_list_path,
        voc12_root,
        scales,
        inter_transform=None,
        unit=1,
        use_se=False,
        se_path=None,
    ):
        super().__init__(img_name_list_path, voc12_root, transform=None)
        self.scales = scales
        self.unit = unit
        self.inter_transform = inter_transform
        self.use_se = use_se
        self.se_path = se_path

    def __getitem__(self, idx):
        pack = super().__getitem__(idx)

        name = pack["name"]
        img = pack["img"]
        label = pack["label"]

        rounded_size = (
            int(round(img.size[0] / self.unit) * self.unit),
            int(round(img.size[1] / self.unit) * self.unit),
        )

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0] * s), round(rounded_size[1] * s))
            s_img = img.resize(target_size, resample=Image.CUBIC)
            ms_img_list.append(s_img)

        if self.inter_transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.inter_transform(ms_img_list[i])

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(np.flip(ms_img_list[i], -1).copy())

        pack = {}
        pack["img"] = msf_img_list
        pack["label"] = label
        pack["name"] = name

        if self.use_se:
            se = np.load(self.se_path + "/" + name + ".npy")
            se = (se + 1).astype(np.uint8)
            se = torch.from_numpy(se)
            se_flip = torch.flip(se, dims=(1,))

            pack["se"] = [se, se_flip]

        return pack


class VOC12ClsDataset_MyTF(VOC12ImageDataset):
    def __init__(
        self,
        img_name_list_path,
        voc12_root,
        crop,
        resize,
        cj,
        use_se=False,
        se_path=None,
    ):
        super().__init__(img_name_list_path, voc12_root)

        self.label_list = load_image_label_list_from_npy(self.img_name_list)

        self.crop = crop
        self.resize = resize
        self.cj = cj

        self.use_se = use_se
        self.se_path = se_path

        self.tf_rr = imutils.random_resize(self.resize[0], self.resize[1])
        self.tf_rc = imutils.random_crop(self.crop[0])

        self.tf_flip = transforms.RandomHorizontalFlip(p=0.5)
        self.tf_cj = transforms.RandomApply(
            [transforms.ColorJitter(self.cj[0], self.cj[1], self.cj[2], self.cj[3])],
            p=0.8,
        )
        self.tf_gray = transforms.RandomGrayscale(p=0.02)
        self.tf_norm = imutils.normalize()

        self.tf_list = []
        self.tf_list.append(imutils.HWC_to_CHW)
        self.tf_list.append(imutils.torch.from_numpy)
        self.tf_final = transforms.Compose(self.tf_list)

    def apply_tf(self, img):

        img = self.tf_rr(img)
        img = self.tf_flip(img)
        img = self.tf_cj(img)
        img = self.tf_gray(img)

        img = np.asarray(img)
        img = self.tf_norm(img)

        img = self.tf_rc(img)
        img = self.tf_final(img)

        return img

    def apply_tf_se(self, img):

        img = self.tf_rr(img, mode="nearest")
        img = self.tf_flip(img)
        # img = self.tf_cj(img)
        # img = self.tf_gray(img)

        img = np.asarray(img)  # (H,W,3)
        # img = self.tf_norm(img)

        unique_values = np.unique(img)
        sorted_values = np.sort(unique_values)

        mapped_img = np.searchsorted(sorted_values, img)

        img = self.tf_rc(mapped_img)
        img = self.tf_final(img)

        img = img[0].unsqueeze(0)

        return img

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)

        label = torch.from_numpy(self.label_list[idx])

        if self.use_se:

            rng_state = random.getstate()
            rng_state_torch = torch.get_rng_state()
            img = self.apply_tf(img)

            random.setstate(rng_state)
            torch.set_rng_state(rng_state_torch)

            se = np.load(self.se_path + "/" + name + ".npy")

            se = (se + 1).astype(np.uint8)
            se = np.expand_dims(se, axis=2)
            se = np.repeat(se, 3, axis=2)
            se = Image.fromarray(se)
            se = self.apply_tf_se(se)

            pack = {}
            pack["img"] = img
            pack["label"] = label
            pack["name"] = name
            pack["se"] = se
            return pack

        else:
            img = self.apply_tf(img)

            pack = {}
            pack["img"] = img
            pack["label"] = label
            pack["name"] = name

            return pack

    def __len__(self):
        return len(self.img_name_list)
