import numpy as np
import os

# define the directory of the file to write the directory of image and its
# corresponding index
file_dir = "/root/hdd/yankun/Kaggle/quick_draw/all"
file_name = "meta_file"
file_name2 = "meta_file_val"
file_name3 = "image_num"
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
                image_to_num = image_fold_dir + ' ' + str(image_num) + '\n'
                mf3.write(image_to_num)
                image_fold_dir = os.path.join(image_root_dir, image_fold_dir)
                images = os.listdir(image_fold_dir)
                len_images = len(images)
                read_num = min(11000, len_images - 2000)
                read_num = read_num + 2000
                # test_num = 0
                judge_num1 = 0
                judge_num2 = 0

                for image in images:
                    image_dir = os.path.join(image_fold_dir, image)
                    # print(image_dir)
                    result_temp = image_dir + ' ' + str(image_num) + '\n'
                    if judge_num1 < 2000:
                        mf2.write(result_temp)
                        judge_num1 += 1
                    else:
                        mf.write(result_temp)

                    if judge_num2 < read_num:
                        judge_num2 += 1
                    else:
                        break
                    # test_num = test_num + 1
                    # if test_num >= 20:
                    #     break
                image_num = image_num + 1

print("the work is done!!!!!")
