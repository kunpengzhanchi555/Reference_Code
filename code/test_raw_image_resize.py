import numpy as np
import pandas as pd
import os
from pandas import Series, DataFrame
from PIL import Image, ImageDraw

image_size = 224

image_in_path_all = "/root/hdd/yankun/Kaggle/quick_draw/all/test_raw_image"
image_in_all_path = os.listdir(image_in_path_all)
for image_in in image_in_all_path:
    print("the image {} is handling...".format(image_in))
    image_in_path = os.path.join(image_in_path_all, image_in)
    try:
        with Image.open(image_in_path) as image_in_handle:
            image_size_temp = image_in_handle.size
            if image_size_temp[0] == image_size_temp[1] == 224:
                print("already the right")
                continue
            image_out = image_in_handle.resize(
                (image_size, image_size), Image.ANTIALIAS)
            image_out.save(image_in_path)
    except Image.DecompressionBombError as e:
        print("open image {}, some wrongs happen.".format(image_in))
    finally:
        print("the image {} is done".format(image_in))
