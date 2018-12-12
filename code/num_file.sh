#! /bin/sh

for file in $(ls /root/hdd/yankun/Kaggle/quick_draw/all/train_raw)
do 
  echo ${file} >> /root/hdd/yankun/Kaggle/quick_draw/all/num_file
  wc -l /root/hdd/yankun/Kaggle/quick_draw/all/train_raw/${file} >> num_file
done
