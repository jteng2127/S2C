from __future__ import division
from __future__ import print_function

import os
import os.path as osp
from tqdm import tqdm
import pdb
import random
import importlib
import argparse
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from matplotlib import pyplot as plt
import pickle

# Custom
import tools.utils as utils
from tools.imutils import *
from evaluation import eval_in_script

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything.predictor import SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.utils.amg import build_all_layer_point_grids, batch_iterator

import torch.nn.functional as F

import glob
import pdb
import tqdm

# import debugpy
# debugpy.listen(("0.0.0.0", 5678))
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()  # Pause execution until debugger is attached

from dotenv import load_dotenv

load_dotenv()

root_path = "."

sam_path = os.path.join(os.getenv("PRETRAINED_DIR"), "sam_vit_h.pth")
sam = sam_model_registry["vit_h"](checkpoint=sam_path)
sam = sam.to("cuda")

mask_generator = SamAutomaticMaskGenerator(sam)

img_path = os.path.join(os.getenv("VOC2012_DIR"), "JPEGImages")
print(f"img_path: {img_path}")
save_path = os.getenv("SE_DIR")
print(f"save_path: {save_path}")
os.makedirs(save_path, exist_ok=True)

img_list_path = os.path.join(root_path, "voc12", "train_aug.txt")
img_gt_name_list = open(img_list_path).read().splitlines()
img_name_list = [img_gt_name.split(" ")[0][-15:-4] for img_gt_name in img_gt_name_list]


for name in tqdm.tqdm(img_name_list):
    img = plt.imread(os.path.join(img_path, name + ".jpg"))
    masks = mask_generator.generate(img)

    temp = np.full((img.shape[0], img.shape[1]), -1, dtype=int)
    for i, mask in enumerate(reversed(masks)):
        temp[mask["segmentation"]] = i

    np.save(os.path.join(save_path, name), temp)
