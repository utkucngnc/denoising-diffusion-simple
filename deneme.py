import torch as th
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from models.dataset import BatteryDataset as BD

ds = BD(transform=transforms.Compose([
    transforms.Resize(64, antialias=True),
]))

dl = th.utils.data.DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

for i in dl:
    print(i.shape)
