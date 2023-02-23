import os

import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils as smp_utils
from pprint import pprint
from torch.utils.data import DataLoader
from torch import nn

from KSDD import KSDDDataset


# download data
root = "KSDD"

# init train, val, test sets
train_dataset = KSDDDataset(root, "train")
valid_dataset = KSDDDataset(root, "valid")
test_dataset = KSDDDataset(root, "test")

print(f"Train size: {len(train_dataset)}")
print(f"Valid size: {len(valid_dataset)}")
print(f"Test size: {len(test_dataset)}")

n_gpu = 0
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=n_gpu)
valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=n_gpu)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=n_gpu)

# model = smp.Unet(encoder_name="resnet34", encoder_depth=5, encoder_weights="imagenet", decoder_use_batchnorm=True,
#                  activation="sigmoid")
model = smp.PAN(encoder_name="resnet34", encoder_weights="imagenet")
model.__name__ = "PAN"
# model = nn.Sequential(nn.Conv2d(3, 512, kernel_size=7, stride=2, padding=3, bias=False))


loss_fn = smp.losses.DiceLoss("binary")
loss_fn.__name__ = "DiceLoss"

metrics = [smp_utils.metrics.IoU(threshold=0.5)]

optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001)])

# for epoch in range(0, 40):
#     for epoch, data in enumerate(train_dataloader, 0):
#         input, mask = data
#         pred = model(input).squeeze()
#
#         loss = loss_fn(pred, mask)
#         print(loss)

# create epoch runners
# it is a simple loop of iterating over dataloader`s samples
train_epoch = smp_utils.train.TrainEpoch(
    model,
    loss=loss_fn,
    metrics=metrics,
    optimizer=optimizer,
    device="cuda",
    verbose=True,
)

valid_epoch = smp_utils.train.ValidEpoch(
    model,
    loss=loss_fn,
    metrics=metrics,
    device="cuda",
    verbose=True,
)

# train model for 40 epochs

max_score = 0

for epoch in range(0, 100):

    print('\nEpoch: {}'.format(epoch))
    train_logs = train_epoch.run(train_dataloader)
    valid_logs = valid_epoch.run(valid_dataloader)

    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, os.path.join('saved_models', 'best_model_{}_{}.pth'.format(model.__name__, epoch)))
        print('Model saved!')

    if epoch == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')
