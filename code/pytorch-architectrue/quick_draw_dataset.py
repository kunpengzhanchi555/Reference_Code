import numpy as np
import pandas as pd
import torch
import torchvision
from pandas import Series, DataFrame
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
import os
import random

global_root_dir = "/root/hdd/yankun/Kaggle/quick_draw/all"
global_meta_file = "meta_file"


class quick_draw_dataset(Dataset):

    def __init__(self, root_dir=global_root_dir, meta_file=global_meta_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.wrong_image_fold = "wrong_image_path_fold"
        self.wrong_image_path_fold = os.path.join(
            root_dir, self.wrong_image_fold)
        meta_file = os.path.join(root_dir, meta_file)
        with open(meta_file) as f:
            lines = f.readlines()
            random.seed(5)
            random.shuffle(lines)
        print("building dataset from %s" % meta_file)
        num = 0
        self.num = len(lines)
        self.metas = []
        for line in lines:
            path, cls = line.rstrip().split()
            self.metas.append((path, int(cls)))
        print("read test_metafile done")

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        path = self.metas[idx][0]
        cls = self.metas[idx][1]
        # transform
        try:
            with Image.open(path) as img:
                img_temp = img.convert('RGB')
                if self.transform is not None:
                    img_temp = self.transform(img_temp)
                return img_temp, cls
        except Image.DecompressionBombError as DBE:
            print("the image of idx is {} wrong1 !!".format(idx))
            path_temp = self.metas[1][0]
            cls_temp = self.metas[1][1]
            with open(self.wrong_image_path_fold, 'a') as f:
                f.write(path)
            with Image.open(path_temp) as img:
                img_temp = img.convert('RGB')
                if self.transform is not None:
                    img_temp = self.transform(img_temp)
                return img_temp, cls_temp
        else:
            print("the image of idx is {} wrong2 !!".format(idx))
