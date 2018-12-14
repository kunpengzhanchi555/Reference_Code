import os
import numpy as np
import csv


npy_path1 = "/root/hdd/yankun/Kaggle/quick_draw/pytorch-architectrue/rst/prob_dense0_1.npy"
npy_path2 = "/root/hdd/yankun/Kaggle/quick_draw/pytorch-architectrue/rst/prob_dense0_2.npy"
npy_path3 = "/root/hdd/yankun/Kaggle/quick_draw/pytorch-architectrue/rst/prob_dense0_3.npy"
result_path = "/root/hdd/yankun/Kaggle/quick_draw/pytorch-architectrue/rst/result.txt"
test_meta_file_path = "/root/hdd/yankun/Kaggle/quick_draw/all/test_meta_file"
# txt_file = open(result_path, 'wb')

print("reading the npy file...")
npy1 = np.load(npy_path1)
npy2 = np.load(npy_path2)
npy3 = np.load(npy_path3)
npy = npy1 + npy2 + npy3
npy_idx = np.argsort(npy, axis=1)
npy_len = npy.shape[0]
print("npy file is done")

print("reading the test_meta_file...")
metas = []
with open(test_meta_file_path) as f:
    for line in f:
        path, cls = line.rstrip().split()
        metas.append(int(cls))
print("test_meta_file is done")

image_map_path = "/root/hdd/yankun/Kaggle/quick_draw/all/image_num_new_new_all"
image_map = []
with open(image_map_path) as image_map_file:
    image_map_lines = image_map_file.readlines()
for line in image_map_lines:
    image_str, image_num = line.rstrip().split()
    image_num = int(image_num)
    image_map.append(image_str)

names = ['key_id', 'word']
submission_path = "/root/hdd/yankun/Kaggle/quick_draw/all/submission.csv"
with open(submission_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(names)
    result_list = []
    for i in range(npy_len):
        print("==> the {}'th image is handling....".format(i))
        npyi_len = npy[i].shape[0]
        str1 = image_map[npy_idx[i][npyi_len - 1]]
        str2 = image_map[npy_idx[i][npyi_len - 2]]
        str3 = image_map[npy_idx[i][npyi_len - 3]]
        str = str1 + ' ' + str2 + ' ' + str3
        im_num = metas[i]
        compent = (im_num, str)
        result_list.append(compent)
    writer.writerows(result_list)
print("done....")
#*********************************************************************
# print npy[0],npy[1]
# print np.where(npy[0] == np.max(npy[0]))[0][0]
# len = npy.shape[0]
# print("the shape is {}".format(npy.shape))
# print("the len is {}".format(len))
# for j in xrange(len):
#     res = int(np.where(npy[j] == np.max(npy[j]))[0][0])
#     txt_file.write(str(res) + '\n')

# print("tyep key is {}".type(key))
# for i in range(5):
#     print("the len of the npy[i] is {}".format(npy[i].shape))
#     print("the type of the npy[i] is {}".format(type(npy[i])))
# txt_file.close()
