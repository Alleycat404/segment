# load best saved checkpoint
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils as smp_utils

from KSDD import KSDDDataset

root = "KSDD"

best_model = torch.load('best_model_Unet_40.pth', map_location="cuda")
# create test dataset
test_dataset = KSDDDataset(root, mode="test")
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)


loss_fn = smp.losses.DiceLoss("binary")
loss_fn.__name__ = "DiceLoss"

metrics = [smp_utils.metrics.IoU(threshold=0.5)]

# evaluate model on test set
test_epoch = smp_utils.train.ValidEpoch(
    model=best_model,
    loss=loss_fn,
    metrics=metrics,
    device="cuda",
)

logs = test_epoch.run(test_dataloader)
print(logs)
