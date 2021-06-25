import torch
import numpy as np
from generator import Generator
from torch.utils.data import DataLoader
from dataset import MapDataset
from torchvision.utils import save_image, make_grid

model = torch.load(config.gen_path)
gen = Generator().to(torch.device('cuda'))
gen.load_state_dict(model)

val_dataset = MapDataset(root_dir = config.val_dir)
val_loader = DataLoader(val_dataset, batch_size = 8, shuffle = True)

x, y = next(iter(val_loader))
x, y = x.to(torch.device('cuda')), y.to(torch.device('cuda'))
gen.eval()

with torch.no_grad():
        
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5
        save_image(y_fake, open(config.save_dir + f'/generated.png', 'wb'))
        save_image(y * 0.5 + 0.5, open(config.save_dir + f'/label.png', 'wb'))
        gen.train()
    
