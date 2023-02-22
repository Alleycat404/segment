# test dataset without transformations for image visualization
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils as smp_utils

from KSDD import KSDDDataset


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        if i <= 1:
            plt.subplot(2, 3, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title(' '.join(name.split('_')).title())
            plt.imshow(image)
        else:
            plt.subplot(2, 3, i + 2)
            plt.xticks([])
            plt.yticks([])
            plt.title(' '.join(name.split('_')).title())
            plt.imshow(image)
    plt.show()


root = "KSDD"

model1 = torch.load("best_model_Unet_40.pth", map_location=torch.device('cpu'))
model2 = torch.load("best_model_FPN_40.pth", map_location=torch.device('cpu'))
model3 = torch.load("best_model_UnetPlusPlus_40.pth", map_location=torch.device('cpu'))

# create test dataset
test_dataset = KSDDDataset(root, mode="test")
test_dataset_vis = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

for data in test_dataset_vis:

    image_vis = data[0].squeeze().detach().cpu().numpy().astype('uint8').transpose(1, 2, 0)
    image, gt_mask = data

    gt_mask = gt_mask.squeeze().detach().cpu().numpy().astype('uint8')

    x_tensor = image.to("cpu")
    pr_mask1 = model1.predict(x_tensor)
    pr_mask1 = (pr_mask1.squeeze().cpu().numpy().round())
    pr_mask2 = model2.predict(x_tensor)
    pr_mask2 = (pr_mask2.squeeze().cpu().numpy().round())
    pr_mask3 = model3.predict(x_tensor)
    pr_mask3 = (pr_mask3.squeeze().cpu().numpy().round())

    visualize(
        image=image_vis,
        ground_truth_mask=gt_mask,
        Unet_40=pr_mask1,
        FPN_40=pr_mask2,
        UnetPlusPlus_40=pr_mask3
    )