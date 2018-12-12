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
test_meta_file = "test_meta_file"


class quick_draw_test_dataset():
    """docstring for quick_draw_test_dataset"""

    def __init__(self, root_dir=root_dir, test_meta_file=test_meta_file, transform=None):
        super(quick_draw_test_dataset, self).__init__()
        self.transform = transform
        test_meta_file_path = os.path.join(root_dir, test_meta_file)
        with open(test_meta_file_path) as f:
            lines = f.readlines()
        print("building test dataset from : {}".format(test_meta_file_path))
        self.num = len(lines)
        self.metas = []
        for line in lines:
            path, cls = line.rstrip().split()
            self.metas.append((path, int(cls)))
        print("read meta done")
        for i in range(10):
            print(self.metas[i][1])


if __name__ == '__main__':
    k = quick_draw_test_dataset()
