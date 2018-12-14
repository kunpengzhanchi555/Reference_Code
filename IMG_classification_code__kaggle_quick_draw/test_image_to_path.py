import numpy as np
import os

file_dir = "/root/hdd/yankun/Kaggle/quick_draw/all"
file_name = "test_meta_file"

test_meta_file_path = os.path.join(file_dir, file_name)
if os.path.exists(test_meta_file_path):
    os.remove(test_meta_file_path)
    print("the file->test_meta_file is removed!,will create it again soon!")

image_root_dir = "/root/hdd/yankun/Kaggle/quick_draw/all/test_raw_image"
image_dirs = os.listdir(image_root_dir)


print("working! please wait:")
# open the file
with open(test_meta_file_path, 'a') as mf:
    for image in image_dirs:
        key_id, _l = image.rstrip().split('.')
        image_dir = os.path.join(image_root_dir, image)
        result_temp = image_dir + ' ' + str(key_id) + '\n'
        mf.write(result_temp)
print("the work is done!!!!!")
