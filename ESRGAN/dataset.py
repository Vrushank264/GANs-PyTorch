import torch
import os
from PIL import Image
from torch.utils.data import DataLoader, dataset
import torchvision.transforms as T
 
class TrainDataset(dataset.Dataset):
    
    def __init__(self,
                 root: str,
                 img_size: int = 128,
                 upscale_factor: int = 4):
        
        super().__init__()
        
        lr_img_size = img_size // upscale_factor
        self.fnames = [os.path.join(root, file) for file in os.listdir(root)]
        
        self.hr_transforms = T.Compose([T.Resize((img_size, img_size), interpolation = Image.BICUBIC),
                                        T.RandomHorizontalFlip(0.25),
                                        T.RandomVerticalFlip(0.25),
                                        T.ToTensor(),
                                        T.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])])
        
        self.lr_transforms = T.Compose([T.ToPILImage(),
                                        T.Resize((lr_img_size, lr_img_size), interpolation = Image.BICUBIC),
                                        T.ToTensor()])
    
    def __len__(self):

      return len(self.fnames) 
        
    def __getitem__(self, idx):
        
        hr = Image.open(self.fnames[idx]).convert('RGB')
        hr = self.hr_transforms(hr)
        lr = self.lr_transforms(hr)
        
        return hr, lr