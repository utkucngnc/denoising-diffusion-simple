from tqdm import tqdm
from models.ddpm import Diffusion_DDPM as Diffusion
from utils import *

diffusion = Diffusion(im_size=64,timestep=1000, function="Linear", device=GetDevice())
model = GetModel("UNet", builtin=False).to(GetDevice())

im = diffusion.Sample(model, n=1)