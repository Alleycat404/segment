import torch
import os
import numpy as np
import PIL.Image as Image
from torch.utils.data import DataLoader
import cv2

class KSDDDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=None):

        assert mode in {"train", "valid", "test"}

        self.root = root
        self.mode = mode
        self.transform = transform

        self.directory = os.path.join(self.root, self.mode)

        self.filenames = self._read_names()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        image_path = os.path.join(self.directory, filename + ".jpg")
        mask_path = os.path.join(self.directory, filename + "_label.bmp")

        image = np.array(cv2.resize(cv2.imread(image_path), (480, 1248)), dtype=np.float32).transpose(2, 0, 1)

        trimap = np.array(cv2.resize(cv2.imread(mask_path, 2), (480, 1248)), dtype=np.float32)
        mask = self._preprocess_mask(trimap)

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask.astype(np.int64)

    def _read_names(self):
        filenames = os.listdir(os.path.join(self.root, self.mode))
        i = 0
        while i < len(filenames):
            if filenames[i][-1] != "g":
                filenames.remove(filenames[i])
            else:
                i += 1
        for i, filename in enumerate(filenames):
            filenames[i] = filename[:-4]
        return filenames


if __name__ == '__main__':
    train_dataset = KSDDDataset("KSDD", "train")
    valid_dataset = KSDDDataset("KSDD", "valid")
    test_dataset = KSDDDataset("KSDD", "test")

    print(f"Train size: {len(train_dataset)}")
    print(f"Valid size: {len(valid_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    n_cpu = os.cpu_count()
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=n_cpu)
    valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=n_cpu)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=n_cpu)