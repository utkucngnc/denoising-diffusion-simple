import torch
import tqdm
from .noise_scheduler import NoiseScheduler

class Diffusion_DDPM:
    def __init__(
                self, 
                im_size = None, 
                timestep = None, 
                function = None,
                device = None
                ) -> None:
        
        self.device = device
        self.timestep = timestep
        self.im_size = im_size
        self.betas = self.GetBetas(function=function)

        assert len(self.betas.shape) == 1, "betas must be 1-D"
        assert (self.betas > 0).all() and (self.betas <= 1).all()

        self.alphas = torch.from_numpy(1 - self.betas).to(self.device)
        self.alpha_hats = torch.cumprod(self.alphas, dim=0).to(self.device)
        
    def GetBetas(self, function = None):
        return NoiseScheduler(function=function, timesteps=self.timestep).Gamma()
    
    def GetNoisedImages(self, img,t, scale = 1.0): # returns timestep, channel, height, width
        
        sqrt_alpha_hats = torch.sqrt(self.alpha_hats[t])[:, None, None, None]
        sqrt_one_minus_alpha_hats = torch.sqrt(1 - self.alpha_hats[t])[:, None, None, None]
        eps = torch.randn_like(img)

        return sqrt_alpha_hats * scale * img + sqrt_one_minus_alpha_hats * eps, eps
    
    def SampleTimesteps(self, n: int):
        return torch.randint(low=1, high=self.timestep, size=(n,))
    
    def Sample(self, model: torch.nn.Module, n: int):
        model.eval()

        with torch.no_grad():
            x = torch.randn((n, 3, self.im_size, self.im_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.timestep)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alphas[t][:, None, None, None]
                alpha_hat = self.alpha_hats[t][:, None, None, None]
                beta = self.betas[t][:, None, None, None]

                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)

        return x



    
