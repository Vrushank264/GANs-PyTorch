import torch
import numpy as np
from generator import Generator
from torch.utils.data import DataLoader
from dataset import MapDataset
from torchvision.utils import save_image, make_grid

model = torch.load('E:/Computer Vision/Pix2pix/Generator.pth')
gen = Generator().to(torch.device('cuda'))
gen.load_state_dict(model)

val_dataset = MapDataset(root_dir = 'E:/Computer Vision/Pix2pix/sketch2Anime/val')
val_loader = DataLoader(val_dataset, batch_size = 4, shuffle = True)

x, y = next(iter(val_loader))
x, y = x.to(torch.device('cuda')), y.to(torch.device('cuda'))
gen.eval()
i = 2
with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5
        i+=1
        save_image(y_fake, open(config.save_dir + f'generated_{i}.png', 'wb'))
        #save_image(x * 0.5 + 0.5, f'E:/Computer Vision/Pix2pix/Results/23rd epoch/input_{i}.png')
        save_image(y * 0.5 + 0.5, open(config.save_dir + f'/label_{i}.png', 'wb'))
        gen.train()
    