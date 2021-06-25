import torch
import numpy as np
import config
import os
from PIL import Image
from torch.utils.data import Dataset

class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        
    def __len__(self):
        return len(self.list_files)
    
    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        ip_image = image[:,512:,:]
        target_image = image[:,:512,:]
        
        aug = config.both_transform(image = ip_image, image0 = target_image)
        ip_image, target_image = aug['image'], aug['image0']
        
        ip_image = config.transform_only_input(image = ip_image)['image']
        target_image = config.transform_only_mask(image = target_image)['image']
        
        return ip_image, target_image
    
x = MapDataset(root_dir = config.root_dir)
print(x.__len__())
