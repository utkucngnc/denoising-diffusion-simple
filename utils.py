from matplotlib import pyplot as plt
from PIL import Image
import torch
from importlib.machinery import SourceFileLoader
import json
import torch.utils.data

def GetConfig(path = './config.json', key = None):
    if key:
        with open(path) as f:
            config = json.load(f)
        if key in config.keys():
            return config[key]
        else:
            raise KeyError(f"{key} not found in config.json")
    else:
        raise NotImplementedError("No key provided to GetConfig()")

def DrawTwoImg(img1, img2): # (1, 256, 256)
    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    axs[0].imshow(img1[0])   
    axs[1].imshow(img2[0]) 
    plt.show()

def Img2Tensor(img, transform = None):
    return torch.reshape(transform(img),(1,)+transform(img).size())

def GetDevice() -> str:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def Extract(input: torch.Tensor, t: torch.Tensor, shape) -> torch.Tensor:
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out

def CreateNoise(shape, noise_fn, device) -> torch.Tensor:
        return noise_fn(*shape,device=device)

def SaveImg(tensor: torch.Tensor, path: str) -> None:
    tensor = tensor.detach().cpu()
    tensor = tensor.permute(1, 2, 0)
    tensor = tensor.numpy()
    tensor = tensor.clip(0, 1)
    tensor = (tensor * 255).astype("uint8")
    img = Image.fromarray(tensor)
    img.save(path)

def GetModel(model_name: str, builtin = None):
    cfg = GetConfig("Models")[model_name]
    model = SourceFileLoader(model_name, cfg["Path"]).load_module()
    if builtin:
        params = CopyDictSlice(cfg, 0, len(cfg))
        model = getattr(model, model_name)(params)
    else:
        model = getattr(model,model_name)()

    return model

def CopyDictSlice(d: dict, start: int, end: int) -> dict:
    return {k: v for k, v in list(d.items())[start:end]}