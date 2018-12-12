import numpy as np
import pandas as pd
import os
from pandas import Series, DataFrame
from PIL import Image, ImageDraw

# sd = {'python': 9000, 'c++': 9001, 'c#': 9002}
# s3 = Series(sd)
# data = {'name': ['google', 'baidu', 'yahoo'],
#         'marks': [100, 200, 300], 'price': [1, 2, 3]}
# f1 = DataFrame(data)
# f2 = DataFrame(data, columns=['name', 'price', 'marks'])
"""**********************************************************************************"""
data_path = "/root/hdd/yankun/Kaggle/quick_draw/all/train_raw"
data_save_path = "/root/hdd/yankun/Kaggle/quick_draw/all/train_raw_image"
filelist = os.listdir(data_path)
for infile in filelist:
    # to replace all the blank to '_'
    os.rename(os.path.join(data_path, infile),
              os.path.join(data_path, infile.replace(' ', '_')))
    image_name = infile.replace(' ', '_')
    image_name = image_name.replace('.csv', '')
    data_save_path_temp = os.path.join(data_save_path, image_name)

    # create the dir for image of this class
    if not os.path.exists(data_save_path_temp):
        os.mkdir(data_save_path_temp)
    else:
        continue
    # to read the data from the file pear.csv
    names = ['countrycode', 'drawing', 'key_id',
             'recognized', 'timestamp', 'word']
    # pear = pd.read_csv(os.path.join(data_path, "pear.csv"), names=names)
    infile_temp = os.path.join(data_path, infile.replace(' ', '_'))
    pear = pd.read_csv(infile_temp, names=names)
    # to read the coordinate and the name of the image
    draws = pear['drawing']
    image_name = pear['word'][1]
    image_name = image_name.replace(' ', '_')

    print(infile, "******************************************** start!!")
    print("please wait......")
    # to gain the number of the image of this class
    len_draws = len(draws)
    # to test the program
    # len_draws = min(len_draws, 3)
    for num_draw in range(len_draws - 1):
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
        new_image_name = image_name + "_" + str(num_draw) + ".jpg"
        # print(new_image_name)
        # print(type(new_image_name))
        new_image_name = os.path.join(data_save_path_temp, new_image_name)
        new_image.save(new_image_name)
        # new_image.save("a_2.jpg")
        new_image.close()
        # image.show()
        image.close()
        del new_image
        del image
    del pear
    print(infile, "******************************* done!!")
