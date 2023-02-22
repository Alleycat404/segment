import os
import random
import shutil

root_dir = 'train'
tar_dir = 'test'

file_number = len(os.listdir(root_dir)) / 2
file_name = os.listdir(root_dir)
for name in file_name:
    if name[-1] == "p":
        file_name.remove(name)

test_number = int(0.1 * file_number)
sample = random.sample(file_name, test_number)

print(len(sample))  # 查看采样结果
for name in sample:  # 遍历采样得到的所有数据
    shutil.move(os.path.join(root_dir, name), os.path.join(tar_dir, name))  # 将原文件夹中采样得到的数据移动到新文件夹中
    shutil.move(os.path.join(root_dir, name[:-4] + "_label.bmp"), os.path.join(tar_dir, name[:-4] + "_label.bmp"))

