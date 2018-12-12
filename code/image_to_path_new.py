# import numpy as np
import os
import math
import random

# define the directory of the file to write the directory of image and its corresponding index
file_dir = "/root/hdd/yankun/Kaggle/quick_draw/all"
file_name = "meta_file_new"
file_name2 = "meta_file_val_new"
file_name3 = "image_num_new"
# define the directory of the whole image folders
image_root_dir = "/root/hdd/yankun/Kaggle/quick_draw/all/train_raw_image"
image_fold_dirs = os.listdir(image_root_dir)
# print(image_fold_dirs)
# to traverse the entire directory of folders for class of images
image_num = 0
meta_file_path = os.path.join(file_dir, file_name)
meta_file2_path = os.path.join(file_dir, file_name2)
meta_file3_path = os.path.join(file_dir, file_name3)
if os.path.exists(meta_file_path):
    os.remove(meta_file_path)
    print("the file->meta_file is removed!,will create it again soon!")
if os.path.exists(meta_file2_path):
    os.remove(meta_file2_path)
    print("the file->meta_file_val is removed, will create it again soon!")
if os.path.exists(meta_file3_path):
    os.remove(meta_file3_path)
    print("the file->image_num is removed, will create it again soon!")

print("working! please wait:")
# open the file
with open(meta_file_path, 'a') as mf:
    with open(meta_file2_path, 'a') as mf2:
        with open(meta_file3_path, 'a') as mf3:
            # to traverse all the images,and transpose it to binary of path and
            # index
            for image_fold_dir in image_fold_dirs:
                print("the image of {} is creating path....".format(image_fold_dir))
                # to record the class_number of each image
                image_to_num = image_fold_dir + ' ' + str(image_num) + '\n'
                mf3.write(image_to_num)
                # to handle each class of images
                image_fold_dir = os.path.join(image_root_dir, image_fold_dir)
                images = os.listdir(image_fold_dir)
                len_images = len(images)

                val_len = math.floor(len_images * 0.1)
                test_len = len_images - val_len

                all_lines = []
                test_lines = []
                val_lines = []

                for image in images:
                    image_dir = os.path.join(image_fold_dir, image)
                    result_temp = image_dir + ' ' + str(image_num) + '\n'
                    all_lines.append(result_temp)
                random.shuffle(all_lines)
                judge_num = 0
                for line in all_lines:
                    if judge_num >= val_len:
                        test_lines.append(line)
                        judge_num += 1
                    else:
                        val_lines.append(line)
                        judge_num += 1

                mf.writelines(test_lines)
                mf2.writelines(val_lines)

                image_num = image_num + 1

print("the work is done!!!!!")
