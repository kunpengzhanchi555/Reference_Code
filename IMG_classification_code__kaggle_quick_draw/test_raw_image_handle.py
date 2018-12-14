import numpy as np
import pandas as pd
import os
from pandas import Series, DataFrame
from PIL import Image, ImageDraw

data_path = "/root/hdd/yankun/Kaggle/quick_draw/all/test_raw.csv"
data_save_path = "/root/hdd/yankun/Kaggle/quick_draw/all/test_raw_image"

if not os.path.exists(data_path):
    os.mkdir(data_save_path)
else:
    pass
names = ['key_id', 'countrycode', 'drawing']
# read information from the test_raw.csv
print("reading the file : {}, please wait...".format("test_raw.csv"))
test_raw_file = pd.read_csv(data_path, names=names)

draws = test_raw_file['drawing']
image_id = test_raw_file['key_id']

len_draws = len(draws)
print("reading file is done")
print("the image is transforming, please wait......")
for num_draw in range(len_draws - 1):
    print("the {} image is transforming..".format(num_draw))
    b = eval(draws[num_draw + 1])
    x_max = y_max = x_min = y_min = 0
    # find the biggest distance for x and y
    for k in range(len(b)):
        np_b = np.array(b[k])
        xx_max = np_b[0, :].max()
        xx_min = np_b[0, :].min()
        yy_max = np_b[1, :].max()
        yy_min = np_b[1, :].min()
        if k == 0:
            x_max = xx_max
            y_max = yy_max
            x_min = xx_min
            y_min = yy_min
        else:
            x_max = max(x_max, xx_max)
            y_max = max(y_max, yy_max)
            x_min = min(x_min, xx_min)
            y_min = min(y_min, yy_min)

    # w = x_max - x_min
    # h = y_max - y_min
    # s = max(w, h)
    image_size = max(x_max - x_min, y_max - y_min)
    image_size = image_size + 10
    # image_size = 500
    image_size = int(image_size)
    image = Image.new("P", (image_size, image_size), color=255)
    image_draw = ImageDraw.Draw(image)
    # to normalize all the point
    for k in range(len(b)):
        np_b = np.array(b[k])
        # np_b[0, :] = (np_b[0, :] - x_min) / s
        # np_b[1, :] = (np_b[1, :] - y_min) / s

        np_b[0, :] = np_b[0, :] - x_min + 5
        np_b[1, :] = np_b[1, :] - y_min + 5

        len_np_b = np_b[0].shape[0]
        for i in range(len_np_b - 1):
            image_draw.line([np_b[0][i], np_b[1][i],
                             np_b[0][i + 1], np_b[1][i + 1]], fill=0)
    new_image = image.convert('RGB')
    # print(new_image.mode)
    # new_image.show()
    new_image_name = image_id[num_draw + 1] + ".jpg"
    # print(new_image_name)
    # print(type(new_image_name))
    new_image_name = os.path.join(data_save_path, new_image_name)
    new_image.save(new_image_name)
    # new_image.save("a_2.jpg")
    new_image.close()
    # image.show()
    image.close()
    del new_image
    del image
    print("the {} image transforming is done".format(num_draw))

print("the test_raw file is done!")
