import numpy as np
import pandas as pd
import torch
import torchvision
from pandas import Series, DataFrame
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
import os
import random

root_dir = "/root/hdd/yankun/Kaggle/quick_draw/all"
meta_file = "meta_file"
meta_file_val = "meta_file_val"


meta_file = os.path.join(root_dir, meta_file)
meta_file_val = os.path.join(root_dir, meta_file_val)
with open(meta_file) as f:
    lines = f.readlines()
    print(type(lines))
    num = 0

    for line in lines:
        print(line)
        num += 1
        if num >= 10:
            break

    random.seed(5)
    random.shuffle(lines)
    num = 0
    for line in lines:
        print(line)
        num += 1
        if num >= 10:
            break
