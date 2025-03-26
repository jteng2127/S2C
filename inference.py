from __future__ import division
from __future__ import print_function

from dotenv import load_dotenv

load_dotenv()

import os
import os.path as osp
from tqdm import tqdm
import pdb
import random
import importlib
import argparse
import logging
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from matplotlib import pyplot as plt

# Custom
import tools.utils as utils
from evaluation import eval_in_script

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epoch", default=0, type=int)

    # Dataset
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/train.txt", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--use_se", action="store_true")
    parser.add_argument("--se_path", default=os.getenv("SE_DIR"), type=str)

    # Augmentation
    parser.add_argument("--resize", default=[256, 512], nargs="+", type=float)
    parser.add_argument("--crop", default=256, type=int)
    parser.add_argument("--cj", default=[0.4, 0.4, 0.4, 0.1], nargs="+", type=float)

    # Attributes
    parser.add_argument("--C", default=20, type=int)
    parser.add_argument("--D", default=256, type=int)
    parser.add_argument("--th_multi", default=0.5, type=float)
    parser.add_argument("--W", default=[1.0, 1.0, 1.0], nargs="+", type=float)
    parser.add_argument("--T", default=1, type=float)

    # Policy
    parser.add_argument("--lr", default=0.02, type=float)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--max_epochs", default=40, type=int)
    parser.add_argument("--sstart", default=2, type=int)

    # Experiments
    parser.add_argument("--model", default="vanilla", type=str)
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--seed", default=5123, type=int)

    # Output
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--dict", action="store_false")
    parser.add_argument("--crf", action="store_true")
    parser.add_argument("--print_freq", default=100, type=int)
    parser.add_argument("--out_num", default=100, type=int)
    parser.add_argument("--alphas", default=[6, 10, 24], nargs="+", type=int)

    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    # args.name = time.strftime("%y%m%d") + '_' + args.model + '_' + args.name
    print(args.name)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    (
        exp_path,
        ckpt_path,
        train_path,
        val_path,
        infer_path,
        dict_path,
        crf_path,
        log_path,
    ) = utils.make_path_with_log(args)

    # Logger
    if osp.isfile(log_path):
        os.remove(log_path)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler(log_path)
    logger.addHandler(file_handler)

    # Tensorboard
    # writer = SummaryWriter(log_dir=exp_path)

    logger.info("-" * 52 + " SETUP " + "-" * 52)
    for arg in vars(args):
        logger.info(arg + " : " + str(getattr(args, arg)))
    logger.info("-" * 111)

    logger.info("Start experiment " + args.name + "!")

    # Load Dataset
    train_dataset = utils.build_dataset_sam(
        args, phase="train", path=args.train_list, use_se=True, se_path=args.se_path
    )
    val_dataset = utils.build_dataset_sam(
        args, phase="val", path=args.val_list, use_se=True, se_path=args.se_path
    )

    # train_dataset = Subset(train_dataset, range(0, 50))
    val_dataset = Subset(val_dataset, range(500, len(val_dataset)))

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=4,
    )
    val_data_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True)

    logger.info("Train dataset is loaded from " + args.train_list)
    logger.info("Validation dataset is loaded from " + args.val_list)

    # Check image count, batch count, and max step
    train_num_img = len(train_dataset)
    train_num_batch = len(train_data_loader)
    max_step = train_num_batch * args.max_epochs
    args.max_step = max_step

    # Load Model
    model = getattr(
        importlib.import_module("models.model_" + args.model), "model_WSSS"
    )(args, logger=logger)
    model.load_model(args.epoch, ckpt_path)

    model.train_setup()
    model.set_phase("eval")
    model.infer_init()

    print(f"vis_path: {val_path}")
    print(f"dict_path: {dict_path}")
    print(f"crf_path: {crf_path}")
    for iter, pack in enumerate(tqdm(val_data_loader)):
        model.unpack(pack)
        model.infer_multi(
            args.epoch,
            val_path,
            dict_path,
            crf_path,
            vis=args.vis,
            dict=args.dict,
            crf=args.crf,
        )

    eval_dict = eval_in_script(logger=logger, eval_list="train", pred_dir=dict_path)

    th_temp = eval_dict["th"]
    miou_temp = eval_dict["miou"]
    mp_temp = eval_dict["mp"]
    mr_temp = eval_dict["mr"]

    epo_str = str(args.epoch).zfill(3)
    miou_temp_str = str(round(miou_temp, 3))
    th_temp_str = str(round(th_temp, 3))
    logger.info(
        "Epoch " + epo_str + " max miou : " + miou_temp_str + " at " + th_temp_str
    )
