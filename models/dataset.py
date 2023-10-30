import torch
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms

from utils import GetConfig

class BatteryDataset(Dataset):
    def __init__(self, transform = None) -> None:
        self.cf = GetConfig(key="Dataset")
        self.root_dir = self.cf["Path"]
        self.im_size = self.cf["Image Size"]
        self.imgs = io.imread(self.root_dir)
        self.transform = transform
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        sample = transforms.ToTensor()(self.imgs[index])
        if self.transform:
            sample = self.transform(sample)
        #return torch.cat([sample, sample, sample], dim=0)
        return sample
        
        