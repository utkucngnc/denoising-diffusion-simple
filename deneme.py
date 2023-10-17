import torch as th
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from models.dataset import BatteryDataset as BD
from models.noise_scheduler import NoiseScheduler as NS

betas = NS(function="Sigmoid", timesteps=10).betas

print(betas, betas.dtype, betas.shape)