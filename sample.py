from models.ddpm import Diffusion_DDPM as Diffusion
from utils import *
import torch.nn as nn
import torch.optim as optim
from models.dataset import BatteryDataset as BD
from torch.utils.data import DataLoader
from torchvision import transforms
import logging
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

def eval(args):
    device = GetDevice()
    hp = GetConfig(key="Hyperparameters")
    diffusion = Diffusion(im_size=hp["Image Size"],timestep=hp["Timestep"], function=hp["Noise Function"], device=device)
    model = GetModel("UNet", builtin=False).to(device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    ckpt = torch.load(os.path.join("models", args.run_name, "ckpt.pt"))
    model.load_state_dict(ckpt['model_state_dict'])
    
    sampled_images = diffusion.Sample(model, n=1)
    SaveImg(sampled_images, os.path.join("results", args.run_name, "sample.jpg"))       
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="DDPM", help="Name of the run")
    args = parser.parse_args()
    eval(args)