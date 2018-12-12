import numpy as np
import pandas as pd
import os
from pandas import Series, DataFrame
from PIL import Image, ImageDraw


image_in_folder_root_path = "/root/hdd/yankun/Kaggle/quick_draw/all/train_raw_image"
image_in_folder_path = os.listdir(image_in_folder_root_path)
wrong_image_fold = "wrong_image_resize_path_fold"
wrong_image_path_fold = os.path.join(
    "/root/hdd/yankun/Kaggle/quick_draw/all", wrong_image_fold)
image_size = 256
num = 1
for path in image_in_folder_path:
    print("index:{}->the image of calss:{} is handling...".format(num, path))
    image_in_path_all = os.path.join(image_in_folder_root_path, path)
    image_in_all_path = os.listdir(image_in_path_all)
    for image_in in image_in_all_path:
        image_in_path = os.path.join(image_in_path_all, image_in)
        try:
            with Image.open(image_in_path) as image_in_handle:
                image_out = image_in_handle.resize(
                    (image_size, image_size), Image.ANTIALIAS)
                image_out.save(image_in_path)
        except Image.DecompressionBombError as e:
            with open(wrong_image_path_fold, 'a') as f:
                f.write(image_in)
    num += 1
    print("the image of calss:{} handle is done")
