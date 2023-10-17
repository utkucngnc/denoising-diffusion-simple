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

def train(args):
    device = GetDevice()
    hp = GetConfig("Hyperparameters")

    ds = BD(transform = transforms.Compose([
        transforms.Resize(hp["Image Size"],antialias=True),
    ]))
    
    dataloader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    model = GetModel("UNet", builtin=False).to(device)
    model = model.double()
    optimizer = optim.AdamW(model.parameters(), lr=hp["Learning Rate"])
    mse = nn.MSELoss()
    diffusion = Diffusion(im_size=hp["Image Size"],timestep=hp["Timestep"], function=hp["Noise Function"], device=device)
    l = len(dataloader)

    for epoch in range(hp["Epochs"]):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, images in enumerate(pbar):
            images = images.to(device).double()
            t = diffusion.SampleTimesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.GetNoisedImages(images, t)
            predicted_noise = model(x_t, t.double())
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        sampled_images = diffusion.Sample(model, n=images.shape[0])
        SaveImg(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="DDPM", help="Name of the run")
    args = parser.parse_args()
    train(args)