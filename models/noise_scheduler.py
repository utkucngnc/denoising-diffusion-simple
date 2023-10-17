import math
import numpy as np

from utils import GetConfig

### Add logging to this class

### Add documentation to this class

class NoiseScheduler:
    def __init__(self, function = None, timesteps = None) -> None:
        self.timesteps = timesteps
        self.t = np.linspace(0,1,timesteps)

        if function not in ["Cosine", "Sigmoid"]:
            self.function = "Linear"
        else:
            self.function = function
    
    def Gamma(self):
        if self.function == "Linear":
            scale = 1000 / self.timesteps # 1000 is the default value for beta in the paper
            beta_start, beta_end = GetConfig("Noise Function")[self.function].values()
            beta_end = beta_end * scale
            beta_start = beta_start * scale

            return np.linspace(
                beta_start, beta_end, self.timesteps, dtype=np.float64
                )
        
        elif self.function == "Cosine":
            start, end, tau = GetConfig("Noise Function")[self.function].values()
            v_start = np.cos(start * math.pi / 2) ** (2 * tau)
            v_end = np.cos(end * math.pi / 2) ** (2 * tau)
            output = np.cos((self.t * (end - start) + start) * math.pi / 2) ** (2 * tau)
            output = (v_end - output) / (v_end - v_start)

            return output
        
        elif self.function == "Sigmoid":
            start, end, tau = GetConfig("Noise Function")[self.function].values()
            sigmoid = lambda x: 1 / (1 + np.exp(-x))
            v_start = sigmoid(start / tau)
            v_end = sigmoid(end / tau)
            output = sigmoid((self.t * (end - start) + start) / tau)
            output = (v_end - output) / (v_end - v_start)

            return output