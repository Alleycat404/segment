import os

root = "."

for file in os.listdir(root):
    if file[-1] == "p" or file[-1] == "g":
        os.rename(file, file[1:])
