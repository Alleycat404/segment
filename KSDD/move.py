import os
import shutil

root = "."
for folder in os.listdir(root):
    if os.path.isdir(os.path.join(root, folder)):
        for file in os.listdir(os.path.join(root, folder)):
            shutil.move(os.path.join(root, folder, file), os.path.join(root, folder + file))
